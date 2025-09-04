#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

from transformers import (
    HfArgumentParser,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


# =========================
# Dataclass Args
# =========================

@dataclass
class ModelArguments:
    model_path: str = field(
        default="pretrained_models/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Path or hub ID of the Qwen2.5-VL checkpoint."},
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Attention impl (e.g. 'flash_attention_2' or 'eager')."},
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "auto | bfloat16 | float16 | float32"},
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "Device map for model placement (e.g., 'auto')."},
    )
    max_new_tokens: int = field(
        default=1280,
        metadata={"help": "Generation length for the model."},
    )


@dataclass
class DataArguments:
    nusc_root: str = field(
        metadata={"help": "nuScenes dataroot (folder containing the 1.0 subfolder)."}
    )
    json_path: str = field(
        metadata={"help": "Path to DriveLM-style JSON index of scenes/key_frames."}
    )
    output_json: str = field(
        default="converted_output.json",
        metadata={"help": "Where to save the converted dataset. Auto-suffixed per rank if world_size>1."},
    )
    camera_order: list[str] = field(
        default_factory=lambda: [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ],
        metadata={"help": "Order of surrounding cameras to read per sample."},
    )
    min_pixels: int = field(default=64 * 28 * 28)
    max_pixels: int = field(default=1280 * 28 * 28)
    traj_steps: int = field(default=8)
    num_frames: int = field(default=4, metadata={"help": "Number of consecutive frames to use for temporal sequence."})
    use_multiframe: bool = field(default=True, metadata={"help": "Whether to use multi-frame input (4 frames x 6 cameras)."})
    use_multiturn_dialog: bool = field(default=False, metadata={"help": "Whether to use 4-turn dialog generation instead of single turn."})
    max_entries: int = field(default=500, metadata={"help": "Cap per rank (set -1 for no cap)."})
    # NEW: sharding across ranks
    rank_idx: int = field(default=0, metadata={"help": "This process's rank index [0..world_size-1]."})
    world_size: int = field(default=1, metadata={"help": "Total number of ranks."})


parser = HfArgumentParser((ModelArguments, DataArguments))


# =========================
# Helpers
# =========================

def quaternion_yaw(q: Quaternion) -> float:
    v = np.dot(q.rotation_matrix, np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(v[1], v[0]))


def correct_yaw(yaw: float) -> float:
    return (-np.pi - yaw) if yaw <= 0 else (np.pi - yaw)


def get_pose_info(nusc: NuScenes, sample_token: str, camera: str = "CAM_FRONT"):
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"][camera])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    pos = np.asarray(pose["translation"][:2], dtype=float)
    yaw = quaternion_yaw(Quaternion(pose["rotation"]))
    yaw_corr = correct_yaw(yaw)
    return sample["next"], pos, yaw_corr, sd["filename"]


def get_surrounding_views(nusc: NuScenes, sample_token: str, order: list[str]) -> list[str]:
    sample = nusc.get("sample", sample_token)
    return [nusc.get("sample_data", sample["data"][cam])["filename"] for cam in order]


def get_surrounding_views_multiframe(nusc: NuScenes, sample_token: str, order: list[str], num_frames: int = 4) -> list[list[str]]:
    """
    Get surrounding views for multiple frames (temporal sequence).
    Returns a list of frame lists, where each frame list contains filenames for all cameras in order.
    """
    frames_data = []
    current_token = sample_token
    
    for frame_idx in range(num_frames):
        if not current_token:
            break
            
        sample = nusc.get("sample", current_token)
        frame_filenames = [nusc.get("sample_data", sample["data"][cam])["filename"] for cam in order]
        frames_data.append(frame_filenames)
        
        # Move to next sample (previous in time for reverse temporal order)
        current_token = sample.get("prev", None)
    
    # Reverse to get chronological order (oldest to newest)
    return list(reversed(frames_data))


def compute_relative_traj(nusc: NuScenes, start_token: str, steps: int) -> list[tuple[float, float]]:
    fut_traj: list[tuple[float, float]] = []
    next_token, start_pos, start_yaw, _ = get_pose_info(nusc, start_token)
    cos_yaw, sin_yaw = np.cos(-start_yaw), np.sin(-start_yaw)
    rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)
    for _ in range(steps):
        if not next_token:
            break
        next_token, pos, _, _ = get_pose_info(nusc, next_token)
        rel_pos = rot_mat @ (pos - start_pos)
        fut_traj.append((float(rel_pos[0]), float(rel_pos[1])))
    return fut_traj


def compute_relative_history_traj(nusc: NuScenes, start_token: str, steps: int) -> list[tuple[float, float]]:
    """Compute relative trajectory for historical waypoints (going backwards in time)."""
    hist_traj: list[tuple[float, float]] = []
    _, start_pos, start_yaw, _ = get_pose_info(nusc, start_token)
    cos_yaw, sin_yaw = np.cos(-start_yaw), np.sin(-start_yaw)
    rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)
    
    # Go backwards to get history
    current_token = start_token
    sample = nusc.get("sample", current_token)
    prev_token = sample.get("prev", None)
    
    for _ in range(steps):
        if not prev_token:
            break
        _, pos, _, _ = get_pose_info(nusc, prev_token)
        rel_pos = rot_mat @ (pos - start_pos)
        hist_traj.append((float(rel_pos[0]), float(rel_pos[1])))
        
        # Move to previous sample
        sample = nusc.get("sample", prev_token)
        prev_token = sample.get("prev", None)
    
    # Reverse to get chronological order (oldest to newest)
    return list(reversed(hist_traj))



def build_image_messages(
    multiframe_paths: list[list[Path]],  # 4 frames x 6 cameras each
    min_pixels: int, 
    max_pixels: int,
    future_waypoints: list[tuple[float, float]],
    history_waypoints: list[tuple[float, float]],
    navigation_info: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Build image messages for multi-frame input (4 frames x 6 cameras = 24 images total).
    
    Args:
        multiframe_paths: List of 4 frames, each containing 6 camera paths in order 
                         [FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT]
    """
    num_frames = len(multiframe_paths)
    if num_frames != 4:
        raise ValueError(f"Expected 4 frames, got {num_frames}")
    
    for frame_idx, frame_paths in enumerate(multiframe_paths):
        if len(frame_paths) != 6:
            raise ValueError(f"Expected 6 cameras in frame {frame_idx}, got {len(frame_paths)}")
    
    # Flatten all images: frame0_cam0, frame0_cam1, ..., frame0_cam5, frame1_cam0, ...
    all_imgs = []
    for frame_paths in multiframe_paths:
        for path in frame_paths:
            all_imgs.append({"type": "image", "image": str(path), "min_pixels": min_pixels, "max_pixels": max_pixels})
    
    # Format waypoints for instruction
    waypoints_str = "; ".join(f"({x:.2f}, {y:.2f})" for x, y in future_waypoints)
    history_waypoints_str = "; ".join(f"({x:.2f}, {y:.2f})" for x, y in history_waypoints)
    
    # Build instruction with ground truth waypoints and navigation info
    instruction_parts = [
        "You are an autonomous driving labeller. Here are 4 consecutive frames of 6 surrounding onboard camera views from ego vehicle. "
        "Each frame contains cameras in order FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT. "
        "The frames are ordered from oldest to newest (temporal sequence).",
        f"\nGround Truth Future Waypoints (next 8 steps): {waypoints_str}",
    ]
    
    if navigation_info:
        instruction_parts.append(f"\nNavigation Information: {navigation_info}")
    
    if history_waypoints:
        instruction_parts.append(f"\nGround Truth History Waypoints (previous 8 steps): {history_waypoints_str}")
    
    instruction_parts.extend([
        "\nDescribe the Scene in order:",
        "1. Scene Description: describe the scene around the car, analyzing the temporal changes across the 4 frames.",
        "2. Critical Object Description: Imagine you are driving the car. What other road users should you pay "
        "attention to? List two or three of them, specify their location and describe how they move across frames.",
        "3. Reasoning on Intent: describe the likely intent of the road users from (2), what they are doing, and whether it will affect you. "
        "Consider their movement patterns observed across the frames.",
        "4. Trajectory Analysis: Given the ground truth future waypoints provided above, analyze how the planned "
        "trajectory aligns with the scene and navigation requirements. Explain why this trajectory makes sense "
        "given the current traffic situation, road conditions, and temporal dynamics observed in the frames."
    ])
    
    instruction = " ".join(instruction_parts)
    
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [*all_imgs, {"type": "text", "text": instruction}]},
    ]


def to_traj_string(traj: list[tuple[float, float]]) -> str:
    return "; ".join(f"({x:.2f}, {y:.2f})" for x, y in traj)


def generate_multiturn_dialog(
    model,
    processor,
    multiframe_paths: list[list[Path]],
    min_pixels: int,
    max_pixels: int,
    future_waypoints: list[tuple[float, float]],
    history_waypoints: list[tuple[float, float]],
    navigation_info: Optional[str] = None,
    max_new_tokens: int = 320
) -> str:
    """
    Generate text through 4-turn dialog process.
    Each turn focuses on a specific aspect of the driving analysis.
    
    Returns the final combined analysis text.
    """
    
    # Prepare base image content
    all_imgs = []
    for frame_paths in multiframe_paths:
        for path in frame_paths:
            all_imgs.append({"type": "image", "image": str(path), "min_pixels": min_pixels, "max_pixels": max_pixels})
    
    # Format waypoints
    waypoints_str = "; ".join(f"({x:.2f}, {y:.2f})" for x, y in future_waypoints)
    history_waypoints_str = "; ".join(f"({x:.2f}, {y:.2f})" for x, y in history_waypoints)
    
    # Base context for all turns
    base_context = [
        "You are an autonomous driving analyst. Here are 4 consecutive frames of 6 surrounding onboard camera views from ego vehicle. "
        "Each frame contains cameras in order FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT. "
        "The frames are ordered from oldest to newest (temporal sequence).",
        f"\nGround Truth Future Waypoints (next 8 steps): {waypoints_str}",
    ]
    
    if navigation_info:
        base_context.append(f"\nNavigation Information: {navigation_info}")
    
    if history_waypoints:
        base_context.append(f"\nGround Truth History Waypoints (previous 8 steps): {history_waypoints_str}")
    
    base_context_str = " ".join(base_context)
    
    # Define 4 dialog turns
    dialog_turns = [
        {
            "step": 1,
            "prompt": f"{base_context_str}\n\nStep 1 - Scene Description: Analyze and describe the current driving scene around the car. Focus on the road structure, traffic conditions, weather, and overall environment. Describe how the scene changes across the 4 frames.",
            "role": "scene_analyst"
        },
        {
            "step": 2, 
            "prompt": "Step 2 - Critical Object Detection: Based on the scene analysis from Step 1, identify the most critical road users and objects that require attention. List 2-3 of them, specify their locations in the images, and describe their movements across the frames.",
            "role": "object_detector"
        },
        {
            "step": 3,
            "prompt": "Step 3 - Intent Reasoning: Based on the critical objects identified in Step 2, analyze the likely intentions and behaviors of these road users. Consider their movement patterns observed across frames and predict what they might do next. Assess how they might affect the ego vehicle.",
            "role": "intent_analyzer"
        },
        {
            "step": 4,
            "prompt": "Step 4 - Trajectory Analysis: Given all the previous analysis (scene, objects, intents) and the ground truth future waypoints, provide a comprehensive analysis of why the planned trajectory makes sense. Explain how it accounts for the scene conditions, avoids conflicts with other road users, and aligns with navigation requirements.",
            "role": "trajectory_planner"
        }
    ]
    
    # Start conversation history
    conversation_history = [
        {"role": "system", "content": "You are a helpful autonomous driving assistant."}
    ]
    
    # Collect responses from each turn
    turn_responses = []
    
    for turn in dialog_turns:
        # Prepare messages for this turn
        if turn["step"] == 1:
            # First turn includes images
            messages = conversation_history + [
                {"role": "user", "content": [*all_imgs, {"type": "text", "text": turn["prompt"]}]}
            ]
        else:
            # Subsequent turns only include text
            messages = conversation_history + [
                {"role": "user", "content": turn["prompt"]}
            ]
        
        # Generate response for this turn
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        
        # Process vision info (only has effect for first turn with images)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        model_device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            else:
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode response
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": turn["prompt"]})
        conversation_history.append({"role": "assistant", "content": response})
        
        # Store response
        turn_responses.append(f"**{turn['role'].title()}**: {response}")
    
    # Combine all responses into final analysis
    final_analysis = "\n\n".join(turn_responses)
    return final_analysis


def pick_dtype(arg: str) -> torch.dtype:
    if arg == "bfloat16":
        return torch.bfloat16
    if arg == "float16":
        return torch.float16
    if arg == "float32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


# =========================
# Main
# =========================

def main():
    model_args, data_args = parser.parse_args_into_dataclasses()

    rank = int(data_args.rank_idx)
    world = int(data_args.world_size)
    assert 0 <= rank < world, f"rank_idx must be in [0, {world-1}], got {rank}"

    model_path = Path(model_args.model_path)
    nusc_root = Path(data_args.nusc_root)
    json_path = Path(data_args.json_path)
    out_path = Path(data_args.output_json)

    # If running multi-rank and user didn't include a placeholder, auto-suffix.
    if world > 1 and "{rank}" not in out_path.name:
        out_path = out_path.with_name(f"{out_path.stem}.rank{rank}{out_path.suffix}")

    with json_path.open("r") as f:
        index = json.load(f)

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(nusc_root), verbose=True)

    torch_dtype = pick_dtype(model_args.torch_dtype)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=model_args.device_map,
            attn_implementation=model_args.attn_implementation,
        )
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), torch_dtype=torch_dtype, device_map=model_args.device_map
        )

    processor = AutoProcessor.from_pretrained(str(model_path))

    converted_data: list[dict[str, Any]] = []

    # Flat-index sharding without loading all tokens into memory.
    global_idx = 0  # same ordering across ranks
    per_rank_cap = data_args.max_entries if data_args.max_entries >= 0 else None

    scenes = list(index.keys())
    pbar_scenes = tqdm(scenes, desc=f"[rank {rank}/{world}] Scenes", position=0)

    for scene_key in pbar_scenes:
        key_frames = index[scene_key].get("key_frames", [])
        for sample_token in key_frames:
            # shard: only process indices where idx % world_size == rank_idx
            if (global_idx % world) != rank:
                global_idx += 1
                continue

            # per-rank cap
            if per_rank_cap is not None and len(converted_data) >= per_rank_cap:
                break

            fut_traj = compute_relative_traj(nusc, sample_token, steps=data_args.traj_steps)
            if len(fut_traj) < data_args.traj_steps:
                global_idx += 1
                continue

            # For now, navigation_info is None - you can extend this to include actual navigation data
            navigation_info = None  # TODO: Add actual navigation information if available
            
            if data_args.use_multiframe:
                # Get multi-frame data
                multiframe_rel_paths = get_surrounding_views_multiframe(
                    nusc, sample_token, order=data_args.camera_order, num_frames=data_args.num_frames
                )
                
                # Skip if we don't have enough frames
                if len(multiframe_rel_paths) < data_args.num_frames:
                    global_idx += 1
                    continue
                
                # Convert to absolute paths
                multiframe_image_paths = []
                for frame_paths in multiframe_rel_paths:
                    frame_absolute_paths = [nusc_root / rp for rp in frame_paths]
                    multiframe_image_paths.append(frame_absolute_paths)
                
                # Flatten image paths for the final entry
                image_paths = [path for frame_paths in multiframe_image_paths for path in frame_paths]
                
                # Use multi-turn dialog or single turn based on configuration
                if data_args.use_multiturn_dialog:
                    # Compute history trajectory for multi-turn dialog
                    hist_traj = compute_relative_history_traj(nusc, sample_token, steps=data_args.traj_steps)
                    
                    # Generate using 4-turn dialog
                    output_text = generate_multiturn_dialog(
                        model,
                        processor,
                        multiframe_image_paths,
                        data_args.min_pixels,
                        data_args.max_pixels,
                        fut_traj,
                        hist_traj,
                        navigation_info,
                        max_new_tokens=model_args.max_new_tokens // 4  # Divide tokens across 4 turns
                    )
                    output_texts = [output_text]
                else:
                    # Single turn generation
                    hist_traj = compute_relative_history_traj(nusc, sample_token, steps=data_args.traj_steps)
                    image_messages = build_image_messages(
                        multiframe_image_paths,
                        data_args.min_pixels, 
                        data_args.max_pixels,
                        fut_traj,
                        hist_traj,
                        navigation_info
                    )
                    
                    text = processor.apply_chat_template(
                        image_messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
                    )
                    image_inputs, video_inputs = process_vision_info(image_messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    model_device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                    with torch.no_grad():
                        if torch.cuda.is_available():
                            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch_dtype == torch.bfloat16 else torch.float16):
                                generated_ids = model.generate(**inputs, max_new_tokens=model_args.max_new_tokens)
                        else:
                            generated_ids = model.generate(**inputs, max_new_tokens=model_args.max_new_tokens)

                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
                    output_texts = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                
            else:
                # Single frame mode (original behavior) - not supported with multiturn for now
                rel_paths = get_surrounding_views(nusc, sample_token, order=data_args.camera_order)
                image_paths = [nusc_root / rp for rp in rel_paths]
                
                # For single frame, we'll use empty history
                hist_traj = []
                image_messages = build_image_messages(
                    [image_paths],  # Wrap in list to match multiframe format 
                    data_args.min_pixels, 
                    data_args.max_pixels,
                    fut_traj,
                    hist_traj,
                    navigation_info
                )
                
                text = processor.apply_chat_template(
                    image_messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
                )
                image_inputs, video_inputs = process_vision_info(image_messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                model_device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch_dtype == torch.bfloat16 else torch.float16):
                            generated_ids = model.generate(**inputs, max_new_tokens=model_args.max_new_tokens)
                    else:
                        generated_ids = model.generate(**inputs, max_new_tokens=model_args.max_new_tokens)

                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            if data_args.use_multiframe:
                # Multi-frame conversation format
                image_placeholders = ""
                for frame_idx in range(data_args.num_frames):
                    image_placeholders += f"Frame {frame_idx + 1}:\n"
                    for cam_name in ["Front", "Front Left", "Front Right", "Back", "Back Left", "Back Right"]:
                        image_placeholders += f"{cam_name} camera:<image>\n"
                    if frame_idx < data_args.num_frames - 1:
                        image_placeholders += "\n"
                
                human_value = (
                    f"Here are {data_args.num_frames} consecutive frames of 6 surrounding onboard camera views from a vehicle:\n"
                    f"{image_placeholders}"
                    "Predict the optimal driving action for the next 4 seconds with 8 new waypoints and "
                    "use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. "
                    "Consider the temporal dynamics observed across the frames."
                )
            else:
                # Single frame conversation format (original)
                human_value = (
                    "Here are 6 surrounding onboard camera views from a vehicle:\n"
                    "Front camera:<image>\nFront Left camera:<image>\nFront Right camera:<image>\n"
                    "Back camera:<image>\nBack Left camera:<image>\nBack Right camera:<image>\n"
                    "Predict the optimal driving action for the next 4 seconds with 8 new waypoints and "
                    "use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action."
                )

            entry = {
                "datasource": "driveLM",
                "id": str(uuid.uuid4()),
                "image": [str(p) for p in image_paths],
                "conversations": [
                    {
                        "from": "human",
                        "value": human_value,
                    },
                    {
                        "from": "gpt",
                        "value": f"<think>{output_texts[0]}</think><trajectory>{to_traj_string(fut_traj)}</trajectory>",
                    },
                ],
            }
            converted_data.append(entry)

            global_idx += 1  # advance after processing

        # per-rank cap break (outer loop)
        if per_rank_cap is not None and len(converted_data) >= per_rank_cap:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(converted_data, f, indent=2)

    print(
        f"[rank {rank}/{world}] saved {len(converted_data)} entries "
        f"(cap={per_rank_cap}) → {out_path}"
    )


if __name__ == "__main__":
    main()
