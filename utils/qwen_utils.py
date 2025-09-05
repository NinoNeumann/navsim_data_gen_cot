import os
from typing import List, Dict, Any
from PIL import Image
import torch
import numpy as np
import navsim.common.file_ops as fops 
from utils.vis_utils import save_text
from utils.qwen_utils_v2 import summarize_across_views_v2


def resize(img, target_size=28):
    """
    Resize PIL image so that both width and height
    are multiples of target_size (default 28).
    Keeps aspect ratio as much as possible.
    """
    w, h = img.size
    
    # 先按比例缩放，使得最小边 ≥ target_size
    scale = max(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # 再把长宽修正到 target_size 的倍数
    new_w = (new_w // target_size) * target_size
    new_h = (new_h // target_size) * target_size
    
    # 避免被除成 0
    new_w = max(target_size, new_w)
    new_h = max(target_size, new_h)

    img = img.resize((new_w, new_h), Image.BICUBIC)
    return img

def ask_camera_view(
    view_name: str, 
    frame_paths: List[str],
    model,
    processor,
    min_pixels,
    max_pixels,
    max_token_qwen,
) -> Dict[str, str]:
    """
    Ask three concise, multi-turn questions about a specific camera view with N frames.
    Returns: dict with keys: "environment", "key_objects", "decision_notes".
    """

    # ---- 仅用 prompt 约束简短，不做截断 ----
    system_prompt = (
        "You are an intelligent driving vision understanding assistant. Respond briefly and factually.\n"
        "- Keep all answers concise and to-the-point.\n"
        "- Avoid redundancy and flowery adjectives.\n"
        "- Use simple sentences.\n"
        "- Hard length rules (do not exceed):\n"
        "  * Q1 (environment): max 2 short sentences, max ~40 words total.\n"
        "  * Q2 (key objects): max 5 bullets; each bullet <= 8 words; start each bullet with.\n"
        "- Output plain text only (no markdown headings)."
    )

    # Build images
    pil_images = []
    for img_path in frame_paths:
        if not fops.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = fops.image_open(img_path).convert("RGB")
        pil_images.append(img)

    # 像素预算（如需要可在外部传入覆盖）
    min_pixels = 256 * 28 * 28 if min_pixels is None else min_pixels
    max_pixels = 1280 * 28 * 28 if max_pixels is None else max_pixels

    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": f"View: {view_name}. These are frames from this perspective."}]
                + [{"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels} for img in pil_images]
                + [{"type": "text", "text":system_prompt}]
            ),
        },
    ]

    def _generate(current_messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        text = processor.apply_chat_template(current_messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=pil_images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                top_p=0.8,
                temperature=0.2,
                max_new_tokens=max_new_tokens,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        out = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return out.strip()

    # all answer
    env_answer = _generate(messages, max_token_qwen)
    return {
        "answer": env_answer,
    }



def summarize_across_views(
    answers_by_view: Dict[str, Dict[str, str]],
    model,
    processor,
    navigation_info,
    object_position_info,
    max_token_qwen,
) -> str:
    """
    Aggregate eight-view answers and ask model for a final combined response
    to the three questions (environment, key objects, decision notes).
    """
    
    # 构建物体位置信息的描述
    object_position_text = ""
    if object_position_info:
        object_position_text = "\n\nObject positions relative to ego vehicle (BEV perspective):\n"
        for obj in object_position_info:
            name = obj['name']
            rel_pos = obj['relative_position']
            distance = obj['distance']
            heading = obj['heading']
            
            # 将弧度转换为度数，便于理解
            heading_deg = np.degrees(heading)
            
            # 根据相对位置描述方向
            x, y = rel_pos

            # 将坐标保留两位小数
            x_fmt = f"{float(x):.2f}"
            y_fmt = f"{float(y):.2f}"
            pos_str = f"({x_fmt}, {y_fmt})"

            if abs(x) < 2 and abs(y) < 2:
                direction = "very close"
            elif x > 0 and y > 0:
                direction = "front-right"
            elif x > 0 and y < 0:
                direction = "front-left"
            elif x < 0 and y > 0:
                direction = "rear-right"
            elif x < 0 and y < 0:
                direction = "rear-left"
            elif x > 0:
                direction = "front"
            elif x < 0:
                direction = "rear"
            elif y > 0:
                direction = "right"
            else:
                direction = "left"
            
            object_position_text += f"- <obj>{name}</obj>: {direction}, distance {distance:.1f}m, position {pos_str}, heading {heading_deg:.1f}°\n"
    
    system_prompt = system_prompt = f"""
        You are a visual summarization assistant for intelligent driving. You will synthesize answers from EIGHT camera views,
        given the navigation information and object positions.

        Navigation: {navigation_info}
        {object_position_text}

        YOUR OUTPUT MUST FOLLOW EXACTLY THESE THREE MARKDOWN SECTIONS:

        ### 1. Environment Description
        Write ONE brief paragraph that fuses information across all views (no per-view listing).
        Include core traffic context (road type/layout such as lanes/intersections, traffic density/flow, control status like traffic lights/signs,
        special conditions such as weather/visibility/roadworks). Be factual and concise.
        In this paragraph, TAG all traffic-related nouns with <obj></obj>. 
        If a mentioned object exists in the positions list above, include its coordinates INSIDE the tag as <obj>name(x, y)</obj>
        (copy the tuple EXACTLY from the positions list); otherwise tag WITHOUT parentheses.

        ### 2. Key Objects
        Describe briefly ONLY the objects that appear in the positions list above. 
        For each mentioned object, follow the tagging rule <obj>name(x, y)</obj> and add a short factual note (e.g., role/state/risk/direction/proximity).
        Keep it concise and avoid repetition.

        ### 3. Decision
        Make a reasonable driving decision by considering the fused multi-view information and navigation.
        Give ONE brief justification. In this section, TAG all traffic-related nouns with <obj></obj>, and
        if a noun corresponds to an object in the positions list, use <obj>name(x, y)</obj> with the exact coordinates.

        MANDATORY TAGGING RULES (APPLY TO ALL SECTIONS):
        - Objects present in the positions list (i.e., in the "Object positions relative to ego vehicle" above):
        ALWAYS tag as <obj>name(x, y)</obj> whenever mentioned, anywhere in the output.
        Use the canonical 'name' as written in the positions list and copy the 'position' tuple EXACTLY (no rounding, no reordering, no units).
        - Traffic-related nouns NOT in the positions list must still be tagged but WITHOUT coordinates, e.g., <obj>lanes</obj>, <obj>crosswalk</obj>, <obj>traffic light</obj>.
        - NEVER output placeholders like "(x, y)" or "(X, Y)". If coordinates are unavailable, output <obj>name</obj> with NO parentheses.
        - If the same canonical object appears multiple times in the positions list with different coordinates, treat each instance separately and
        tag with its own coordinates when referenced.
        - When an object is repeated within the same bullet/line, include coordinates only at the first mention on that line.

        STYLE:
        - Be concise, factual, and avoid flowery language.
        - Do NOT enumerate per view. Fuse information.
        - Use the exact Markdown headings shown above.
        - Distances/directions may be taken from the provided info (e.g., “front-right”, “6.5m away”), but do NOT invent coordinates.

        SELF-CHECK BEFORE FINALIZING:
        - Every mention of any object that appears in the positions list is tagged as <obj>name(x, y)</obj> with the EXACT tuple from the list.
        - No placeholder coordinates remain. 
        - Traffic-related nouns not in the positions list are tagged as <obj>…</obj> WITHOUT parentheses.
        """


    # Build a single user message that lists per-view answers
    lines = ["The following is a summary input from each perspective:"]
    for view, ans in answers_by_view.items():
        lines.append(f"\n[View: {view}]\n- Environment: {ans.get('environment','')}\n- Key objects: {ans.get('key_objects','')}\n- Decision points: {ans.get('decision_notes','')}")
    lines.append("\nPlease provide a final comprehensive summary based on the above content, retaining only high-value information, and divide it into three clear sections.")


    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": "\n".join(lines)}]},
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.8,
            temperature=0.2,
            max_new_tokens=max_token_qwen,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return out.strip()

def pick_dtype(arg: str) -> torch.dtype:
    if arg == "bfloat16":
        return torch.bfloat16
    if arg == "float16":
        return torch.float16
    if arg == "float32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32



def get_mulit_dialogs(
        model,
        processor,
        camera_order,
        multi_frame_paths,
        navigation_info,
        object_position_info,
        min_pixels,
        max_pixels,
        max_token_qwen:int = 512,
):
    answers_by_view: Dict[str, Dict[str, str]] = {}

    for frame_paths, camera_type in zip(multi_frame_paths, camera_order):
        result = ask_camera_view(
            camera_type, 
            frame_paths, 
            model=model, 
            processor=processor, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels,
            max_token_qwen=max_token_qwen,
        )
        answers_by_view[camera_type] = result
    
        # debug info
        print(f"=======view:{camera_type}=======")
        print(f"description: {result['answer']}")
        print(f"=================================")

    if answers_by_view:
        summary, system_prompt = summarize_across_views_v2(
            answers_by_view, 
            model, 
            processor, 
            navigation_info, 
            object_position_info,
            max_token_qwen,
        )
        print(f"Summary: {summary}")
    
    save_text("./system_prompt.txt", system_prompt)
    
    return summary, answers_by_view

