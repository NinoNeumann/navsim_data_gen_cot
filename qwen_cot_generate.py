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
            torch_dtyp