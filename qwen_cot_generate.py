# from __feature__ import annotation
import json
import os
import sys
from pathlib import Path

from dataclasses import dataclass, field


from typing import Any, Dict, Optional


from utils.navsim_utils import (
    init_scene_loader,
    get_camera_images,
    to_traj_string,
    get_history_future_trajs,
    get_history_navigation_infomation,
    get_object_position

)

from utils.qwen_utils import (
    get_mulit_dialogs,
    pick_dtype
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
    num_hist_frames: int = field(default=4, metadata={"help": "Number of frames to use in the question (should be <= num_hist_traj)"})
    max_scenes: int = field(default=None)
    data_type: str = field(default="mini")

    image_root: str = field(default="./images")
    obs_root: str = field(default="obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data")

    vis_path: str = field(default="./visual")
    if_vis: bool = field(default=False)
    # NEW: sharding across ranks
    rank_idx: int = field(default=0, metadata={"help": "This process's rank index [0..world_size-1]."})
    world_size: int = field(default=1, metadata={"help": "Total number of ranks."})
    max_entries: int = field(default=-1, metadata={"help": "Total number of entries."})


parser = HfArgumentParser((ModelArguments, DataArguments))

# =========================
# Main
# =========================

def main():
    model_args, data_args = parser.parse_args_into_dataclasses()

    # import debugpy
    # debugpy.listen(5678)  # 5678 is port
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()

    rank = int(data_args.rank_idx)
    world = int(data_args.world_size)
    assert 0 <= rank < world, f"rank_idx must be in [0, {world-1}], got {rank}"

    model_path = Path(model_args.model_path)
    out_path = Path(data_args.output_json)

    # processor
    min_pixels = data_args.min_pixels
    max_pixels = data_args.max_pixels

    # navsim
    navsim_root = data_args.nav_root
    num_hist_traj = data_args.num_hist_traj
    num_fut_traj = data_args.num_fut_traj
    num_hist_frames = data_args.num_hist_frames
    data_type = data_args.data_type

    # vis
    vis_path = Path(data_args.vis_path)
    if_vis = data_args.if_vis

    # path prefix
    image_root = Path(data_args.image_root)
    obs_root = data_args.obs_root

    # Validate num_hist_frames
    if num_hist_frames > num_hist_traj:
        print(f"Warning: num_hist_frames ({num_hist_frames}) > num_hist_traj ({num_hist_traj}). Setting num_hist_frames to {num_hist_traj}")
        num_hist_frames = num_hist_traj

    # If running multi-rank and user didn't include a placeholder, auto-suffix.
    if world > 1 and "{rank}" not in out_path.name:
        out_path = out_path.with_name(f"{out_path.stem}.rank{rank}{out_path.suffix}")


    scene_loader = init_scene_loader(
        data_root=navsim_root,
        num_hist_frame=num_hist_traj,
        num_fut_frame=num_fut_traj,
        data_type=data_type
    )

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

    processor = AutoProcessor.from_pretrained(str(model_path), max_pixels=max_pixels, min_pixels=min_pixels)

    converted_data: list[dict[str, Any]] = []

    # Flat-index sharding without loading all tokens into memory.
    global_idx = 0  # same ordering across ranks
    per_rank_cap = data_args.max_entries if data_args.max_entries >= 0 else None

    # tokens = sorted(list(scene_loader.tokens))
    tokens = list(scene_loader.tokens)
    pbar_tokens = tqdm(tokens, desc=f"[rank {rank}/{world}] Scenes", position=0)

    for token in pbar_tokens:
        if (global_idx % world) != rank:
            global_idx += 1
            continue
        
        print(f"==========token:{token}==========")
        scene = scene_loader.get_scene_from_token(token)
        print("scene", scene)
        hist_traj, fut_traj = get_history_future_trajs(scene)
        navigation_info, _ = get_history_navigation_infomation(scene)
        object_position_info = get_object_position(scene)
        # scene_images = get_camera_images(scene, image_root=image_root) # shape:(camera, frames)
        obs_images = get_camera_images(scene, image_root=obs_root, frame_num=num_hist_traj) # shape:(camera, frames)
        image_for_save = get_camera_images(scene, image_root=obs_root, frame_num=num_hist_frames)
        print(obs_images)
        full_image_paths = []
        for camera_images in image_for_save:
            for camera_image in camera_images:
                full_image_paths.append(camera_image)

        output_text, answers_by_view = get_mulit_dialogs(
            model=model,
            processor=processor,
            camera_order=data_args.camera_order,
            multi_frame_paths=obs_images,
            navigation_info=navigation_info,
            object_position_info=object_position_info,

        )

        input_Q = (
            f"Here are {len(data_args.camera_order)} consecutive frames of 6 surrounding onboard camera views from a vehicle:\n"
            f"Front camera: {[f'frame{i}<image>' for i in range(num_hist_frames)]}"
            f"Front Left camera: {[f'frame{i}<image>' for i in range(num_hist_frames)]}"
            f"Front Right camera: {[f'frame{i}<image>' for i in range(num_hist_frames)]}"
            f"Left camera: {[f'frame{i}<image>' for i in range(num_hist_frames)]}"
            f"Right camera: {[f'frame{i}<image>' for i in range(num_hist_frames)]}"
            f"Back camera: {[f'frame{i}<image>' for i in range(num_hist_frames)]}"
            f"Back Left camera: {[f'frame{i}<image>' for i in range(num_hist_frames)]}"
            f"Back Right camera: {[f'frame{i}<image>' for i in range(num_hist_frames)]}"
            f"\nThe navigation information is: {navigation_info}"
            f"\nThe history trajectory is: {to_traj_string(hist_traj)}"
            f"Predict the optimal driving action for the next 4 seconds with 8 new waypoints."
        )
        output_A = (
            f"<think>{output_text}</think><trajectory>{to_traj_string(fut_traj)}</trajectory>"
        )

        entry = {
            "datasource": "navsim",
            "id": str(token),
            "image": [str(p) for p in full_image_paths],
            "conversations": [
                {
                    "from": "human",
                    "value": input_Q,
                },
                {
                    "from": "gpt",
                    "value": output_A,
                }
            ],
        }
        converted_data.append(entry)

        # TODO visualize
        if if_vis:
            vis_path.mkdir(parents=True, exist_ok=True)
            save_text(vis_path / f"{token}.txt", output_A)

        global_idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(converted_data, f, indent=2)

    print(
        f"[rank {rank}/{world}] saved {len(converted_data)} entries "
        f"(cap={per_rank_cap}) → {out_path}"
    )

if __name__ == "__main__":
    main()
