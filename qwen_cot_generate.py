# from __feature__ import annotation
import json
import os
import sys


from typing import Any, Dict, Optional


from utils.navsim_utils import (
    init_scene_loader,
    get_agent_camera_images,
    to_traj_string,

)

from utils.qwen_utils import (
    get_mulit_dialogs
)

from utils.vis_utils import (
    save_text
)

import torch
import numpy as np
from tqdm import tqdm

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
    nav_root: str = field(
        metadata={"help": "navsim dataroot (folder containing the 1.0 subfolder)."}
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
            "CAM_LEFT",
            "CAM_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ],
        metadata={"help": "Order of surrounding cameras to read per sample."},
    )
    min_pixels: int = field(default=64 * 28 * 28)
    max_pixels: int = field(default=1280 * 28 * 28)
    num_hist_traj: int = field(default=4)
    num_fut_traj: int = field(default=6)
    max_scenes: int = field(default=None)
    data_type: str = field(default="mini")

    traj_steps: int = field(default=8)

    vis_path: str = field(default="./visual")
    # NEW: sharding across ranks
    rank_idx: int = field(default=0, metadata={"help": "This process's rank index [0..world_size-1]."})
    world_size: int = field(default=1, metadata={"help": "Total number of ranks."})


parser = HfArgumentParser((ModelArguments, DataArguments))

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

    scene_loader = init_scene_loader()

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
