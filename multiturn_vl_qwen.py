import os
from typing import List, Dict, Any
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_ID = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
DTYPE_STR = os.getenv("TORCH_DTYPE", "bfloat16")  # float16|bfloat16|float32

_model = None
_processor = None


def _get_dtype() -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(DTYPE_STR, torch.bfloat16)


def load_model_and_processor() -> Any:
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    device_map = "auto"
    dtype = _get_dtype()
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).eval()
    return _model, _processor


def ask_camera_view(view_name: str, frame_paths: List[str]) -> Dict[str, str]:
    """
    Ask three multi-turn questions about a specific camera view with 4 frames.

    - view_name: e.g., "front", "front-right", ...
    - frame_paths: list of 4 image file paths for this view

    Returns a dict with keys: "environment", "key_objects", "decision_notes".
    """
    if len(frame_paths) != 4:
        raise ValueError(f"Expected 4 frames for view {view_name}, got {len(frame_paths)}")

    system_prompt = (
        "你是一个智能驾驶视觉理解助手。请根据提供的该视角的四帧图像，"
        "以事实为主进行清晰、简洁的分析。在描述中尽量避免过多形容词。"
    )

    model, processor = load_model_and_processor()

    # Build images
    pil_images = []
    for img_path in frame_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        pil_images.append(Image.open(img_path).convert("RGB"))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": f"视角: {view_name}. 这些是该视角下的4帧画面。"}]
                + [{"type": "image", "image": img} for img in pil_images]
                + [{"type": "text", "text": "问题1：描述车周围的环境，不要使用过多形容词。"}]
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
    messages.append({"role": "user", "content": [{"type": "text", "text": "问题2：描述关键物体。"}]})
    obj_answer = _generate(messages)

    # Q3 follow-up
    messages.append({"role": "assistant", "content": [{"type": "text", "text": obj_answer}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": "问题3：分析该镜头下车辆在做决策时候应该注意些什么。"}]})
    decision_answer = _generate(messages)

    return {
        "environment": env_answer,
        "key_objects": obj_answer,
        "decision_notes": decision_answer,
    }


def summarize_across_views(answers_by_view: Dict[str, Dict[str, str]]) -> str:
    """
    Aggregate six-view answers and ask model for a final combined response
    to the three questions (environment, key objects, decision notes).
    """
    system_prompt = (
        "你是一个智能驾驶领域的视觉总结助手。现在提供来自六个视角的三类回答："
        "(1) 环境描述，(2) 关键物体，(3) 决策注意点。"
        "请将它们进行综合归纳，避免重复与冗余，并输出最终的三个部分：\n"
        "1. 环境描述（简洁、事实为主）\n"
        "2. 关键物体（按类别或优先级凝练）\n"
        "3. 决策注意点（面向驾驶策略的要点清单）"
    )

    # Build a single user message that lists per-view answers
    lines = ["以下是各视角的汇总输入："]
    for view, ans in answers_by_view.items():
        lines.append(f"\n[视角: {view}]\n- 环境: {ans.get('environment','')}\n- 关键物体: {ans.get('key_objects','')}\n- 决策注意点: {ans.get('decision_notes','')}")
    lines.append("\n请基于以上内容输出最终综合总结，仅保留高价值信息，并分成三个清晰小节。")

    model, processor = load_model_and_processor()

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


def main():
    # Example expected directory structure:
    # data/
    #   front/
    #     frame_0.jpg ... frame_3.jpg
    #   front-right/
    #   front-left/
    #   back/
    #   back-right/
    #   back-left/
    base_dir = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))

    camera_to_subdir = {
        "front": "front",
        "front-right": "front-right",
        "front-left": "front-left",
        "back": "back",
        "back-right": "back-right",
        "back-left": "back-left",
    }

    answers_by_view: Dict[str, Dict[str, str]] = {}

    for camera, subdir in camera_to_subdir.items():
        view_dir = os.path.join(base_dir, subdir)
        frame_paths = [os.path.join(view_dir, f"frame_{i}.jpg") for i in range(4)]
        try:
            result = ask_camera_view(camera, frame_paths)
        except Exception as e:
            print(f"[{camera}] 处理失败: {e}")
            continue

        answers_by_view[camera] = result

        print(f"\n===== 视角: {camera} =====")
        print("问题1：环境描述：")
        print(result["environment"]) 
        print("\n问题2：关键物体：")
        print(result["key_objects"]) 
        print("\n问题3：决策注意点：")
        print(result["decision_notes"]) 

    if answers_by_view:
        summary = summarize_across_views(answers_by_view)
        print("\n===== 全视角综合总结 =====")
        print(summary)


if __name__ == "__main__":
    main()


