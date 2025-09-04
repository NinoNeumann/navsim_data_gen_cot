
import os
import torch
from pathlib import Path
import numpy as np
from collections import Counter

from navsim.common.dataclasses import(
    Scene,
    SceneFilter,
    EgoStatus,
    SensorConfig
)

from navsim.common.dataloader import (
    SceneLoader
)
import navsim.common.file_ops as fops 


def init_scene_loader(
    data_root:str,
    num_hist_frame:int,
    num_fut_frame:int,
    max_scenes:int = None,
    data_type:str = "mini",
):
    
    navsim_logs_path = fops.join(data_root,"navsim_logs", data_type)
    navsim_blobs_path = fops.join(data_root,"sensor_blobs", data_type)

    scenes_filter = SceneFilter(
        max_scenes=max_scenes,
        num_history_frames=num_hist_frame,
        num_future_frames=num_fut_frame,
    )
    scenes_loader = SceneLoader(
        data_path=navsim_logs_path,
        original_sensor_path=navsim_blobs_path,
        scene_filter=scenes_filter
    )

    return scenes_loader


def get_camera_images(scene:Scene, image_root:str):
    agentInput = scene.get_agent_input()

    camera_frame_f0_paths = []
    camera_frame_l0_paths = []
    camera_frame_r0_paths = []
    camera_frame_l1_paths = []
    camera_frame_r1_paths = []
    camera_frame_b0_paths = []
    camera_frame_l2_paths = []
    camera_frame_r2_paths = []

    for frame_camera in agentInput.cameras:
        # 根据相机类型添加到对应的路径列表
        camera_frame_f0_paths.append(os.path.join(image_root, frame_camera.cam_f0.camera_path))
        camera_frame_l0_paths.append(os.path.join(image_root, frame_camera.cam_l0.camera_path))
        camera_frame_r0_paths.append(os.path.join(image_root, frame_camera.cam_r0.camera_path))
        camera_frame_l1_paths.append(os.path.join(image_root, frame_camera.cam_l1.camera_path))
        camera_frame_r1_paths.append(os.path.join(image_root, frame_camera.cam_r1.camera_path))
        camera_frame_b0_paths.append(os.path.join(image_root, frame_camera.cam_b0.camera_path))
        camera_frame_l2_paths.append(os.path.join(image_root, frame_camera.cam_l2.camera_path))
        camera_frame_r2_paths.append(os.path.join(image_root, frame_camera.cam_r2.camera_path))

    return [
        camera_frame_f0_paths,
        camera_frame_l0_paths,
        camera_frame_r0_paths,
        camera_frame_l1_paths,
        camera_frame_r1_paths,
        camera_frame_b0_paths,
        camera_frame_l2_paths,
        camera_frame_r2_paths,
    ]

#TODO to finish
def get_object_grounding(scene:Scene):
    start_frame_idx = scene.frames[start_frame_idx]
    current_frame_annotations = current_frame.annotations

    pass

def get_history_navigation_infomation(scene) -> Tuple[str, np.ndarray]:
    agent_input = scene.get_agent_input()
    navigation_command = agent_input.ego_statuses
    all_command = []
    for ego_status in navigation_command:
        cnd = ego_status.driving_command
        all_command.append(tuple(cmd))
    
    if not all_commands:
        default_cmd = np.array([0,0,0,1])
        return "unknown", default_cmd
    
    counter = Counter(all_commands)
    most_common = counter.most_common(1)[0][0]
    most_common_cmd = np.array(most_common)

    direction = map_command_to_direction(most_common_cmd)
    return direction, most_common_cmd


def to_traj_string(traj: list[tuple[float, float]]) -> str:
    return "; ".join(f"({x:.2f}, {y:.2f})" for x, y in traj)


if __name__=="__main__":

    data_root = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data"
    num_hist_frame = 4
    num_fut_frame = 6
    scene_loader = init_scene_loader(data_root, num_hist_frame, num_fut_frame)


    pass