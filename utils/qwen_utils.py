import os
from typing import List, Dict, Any
from PIL import Image
import torch
import numpy as np
from navsim.common.file_ops import image_open




def ask_camera_view(
    view_name: str, 
    frame_paths: List[str],
    model,
    processor,
) -> Dict[str, str]:
    """
    Ask three multi-turn questions about a specific camera view with 4 frames.

    - view_name: e.g., "front", "front-right", ...
    - frame_paths: list of 4 image file paths for this view

    Returns a dict with keys: "environment", "key_objects", "decision_notes".
    """
    if len(frame_paths) != 4:
        raise ValueError(f"Expected 4 frames for view {view_name}, got {len(frame_paths)}")

    system_prompt = (
        "You are an intelligent driving vision understanding assistant. Please analyze the four frames of images "
        "from this perspective based on facts, providing clear and concise analysis. Avoid excessive adjectives in descriptions. "
        "IMPORTANT: When describing traffic-related objects (vehicles, pedestrians, traffic signs, traffic lights, road markings, etc.), "
        "always wrap them with <obj></obj> tags. For example: <obj>car</obj>, <obj>traffic light</obj>, <obj>pedestrian</obj>."
    )

    # Build images
    pil_images = []
    for img_path in frame_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        pil_images.append(image_open(frame_paths).convert("RGB"))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": f"View: {view_name}. These are 4 frames from this perspective."}]
                + [{"type": "image", "image": img} for img in pil_images]
                + [{"type": "text", "text": "Question 1: Describe the environment around the vehicle, avoid using excessive adjectives."}]
            ),
        },
    ]

    def _generate(current_messages: List[Dict[str, Any]], max_new_tokens: int = 512) -> str:
        text = processor.apply_chat_template(current_messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_images], return_tensors="pt").to(model.device)
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

    # Q1
    env_answer = _generate(messages)

    # Q2 follow-up
    messages.append({"role": "assistant", "content": [{"type": "text", "text": env_answer}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Question 2: Describe the key objects."}]})
    obj_answer = _generate(messages)

    # Q3 follow-up
    messages.append({"role": "assistant", "content": [{"type": "text", "text": obj_answer}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Question 3: Analyze what the vehicle should pay attention to when making decisions in this camera view."}]})
    decision_answer = _generate(messages)

    return {
        "environment": env_answer,
        "key_objects": obj_answer,
        "decision_notes": decision_answer,
    }


def summarize_across_views(
    answers_by_view: Dict[str, Dict[str, str]],
    model,
    processor,
    navigation_info,
    object_position_info,
) -> str:
    """
    Aggregate six-view answers and ask model for a final combined response
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
            
            object_position_text += f"- <obj>{name}</obj>: {direction}, distance {distance:.1f}m, heading {heading_deg:.1f}°\n"
    
    system_prompt = (
        "You are a visual summarization assistant in the field of intelligent driving. Now providing answers from six perspectives "
        "in three categories: (1) Environment description, (2) Key objects, (3) Decision points. "
        f"The navigation information is: {navigation_info}"
        f"{object_position_text}"
        "Please synthesize them comprehensively, avoid repetition and redundancy, and output the final three sections:\n"
        "1. Environment description (concise, fact-based)\n"
        "2. Key objects (condensed by category or priority, incorporating position information)\n"
        "3. Decision points (driving strategy-oriented key points, considering object positions and navigation)\n\n"
        "IMPORTANT: When describing traffic-related objects (vehicles, pedestrians, traffic signs, traffic lights, road markings, etc.), "
        "always wrap them with <obj></obj> tags. For example: <obj>car</obj>, <obj>traffic light</obj>, <obj>pedestrian</obj>. "
        "When mentioning object positions, use the relative position information provided above to give spatial context."
    )

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
            max_new_tokens=768,
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
        canera_order,
        multi_frame_paths,
        navigation_info,
        object_position_info,
):
    answers_by_view: Dict[str, Dict[str, str]] = {}

    for frame_paths, camera_type in zip(multi_frame_paths, canera_order):
        result = ask_camera_view(camera_type, frame_paths)
        answers_by_view[camera_type] = result
    
    # debug info
    print(f"=======view:{camera_type}=======")
    print(f"Environment description: {result['environment']}")
    print(f"Key objects: {result['key_objects']}")
    print(f"Decision points: {result['decision_notes']}")
    print(f"=================================")

    if answers_by_view:
        summary = summarize_across_views(answers_by_view, model, processor, navigation_info, object_position_info)
        print(f"Summary: {summary}")
    
    return summary, answers_by_view

