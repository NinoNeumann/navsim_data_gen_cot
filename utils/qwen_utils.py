import os
from typing import List, Dict, Any
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def get_mulit_dialogs():
    pass


def _get_dtype() -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(DTYPE_STR, torch.bfloat16)


def load_model_and_processor(
    model_path:str
) -> Any:
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    device_map = "auto"
    dtype = _get_dtype()
    _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).eval()
    return _model, _processor


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
        pil_images.append(Image.open(img_path).convert("RGB"))

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
    processor
) -> str:
    """
    Aggregate six-view answers and ask model for a final combined response
    to the three questions (environment, key objects, decision notes).
    """
    system_prompt = (
        "You are a visual summarization assistant in the field of intelligent driving. Now providing answers from six perspectives "
        "in three categories: (1) Environment description, (2) Key objects, (3) Decision points. "
        "Please synthesize them comprehensively, avoid repetition and redundancy, and output the final three sections:\n"
        "1. Environment description (concise, fact-based)\n"
        "2. Key objects (condensed by category or priority)\n"
        "3. Decision points (driving strategy-oriented key points)\n\n"
        "IMPORTANT: When describing traffic-related objects (vehicles, pedestrians, traffic signs, traffic lights, road markings, etc.), "
        "always wrap them with <obj></obj> tags. For example: <obj>car</obj>, <obj>traffic light</obj>, <obj>pedestrian</obj>."
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
            print(f"[{camera}] Processing failed: {e}")
            continue

        answers_by_view[camera] = result

        print(f"\n===== View: {camera} =====")
        print("Question 1: Environment description:")
        print(result["environment"]) 
        print("\nQuestion 2: Key objects:")
        print(result["key_objects"]) 
        print("\nQuestion 3: Decision points:")
        print(result["decision_notes"]) 

    if answers_by_view:
        s