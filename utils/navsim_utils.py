
import os
import numpy as np
from collections import Counter
from typing import Tuple

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
        camera_frame_f0_paths.append(fops.join(image_root, frame_camera.cam_f0.camera_path))
        camera_frame_l0_paths.append(fops.join(image_root, frame_camera.cam_l0.camera_path))
        camera_frame_r0_paths.append(fops.join(image_root, frame_camera.cam_r0.camera_path))
        camera_frame_l1_paths.append(fops.join(image_root, frame_camera.cam_l1.camera_path))
        camera_frame_r1_paths.append(fops.join(image_root, frame_camera.cam_r1.camera_path))
        camera_frame_b0_paths.append(fops.join(image_root, frame_camera.cam_b0.camera_path))
        camera_frame_l2_paths.append(fops.join(image_root, frame_camera.cam_l2.camera_path))
        camera_frame_r2_paths.append(fops.join(image_root, frame_camera.cam_r2.camera_path))

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

def get_object_position(scene:Scene):
    start_frame_idx = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[start_frame_idx]

    current_frame_annotations = current_frame.annotations
    object_names = current_frame_annotations.names
    object_boxes = current_frame_annotations.boxes
    
    # 获取自车在当前帧的位置（自车在ego坐标系中位置为(0,0)）
    ego_position = np.array([0.0, 0.0])
    
    # 创建结果列表
    object_info_list = []
    
    for i, (name, box) in enumerate(zip(object_names, object_boxes)):
        # 获取物体在ego坐标系中的位置 (X, Y)
        object_pos = box[:2]  # 取前两个元素作为X, Y坐标
        
        # 计算相对于自车的位置
        relative_position = object_pos - ego_position
        
        # 获取物体的其他信息
        object_length = box[3]  # LENGTH
        object_width = box[4]   # WIDTH
        object_height = box[5]  # HEIGHT
        object_heading = box[6] # HEADING
        
        # 计算距离
        distance = np.linalg.norm(relative_position)
        
        # 创建物体信息字典
        object_info = {
            'name': name,
            'relative_position': relative_position.tolist(),  # [x, y] 相对于自车的位置
            'distance': float(distance),                     # 距离自车的欧几里得距离
            'dimensions': [float(object_length), float(object_width), float(object_height)],  # [长, 宽, 高]
            'heading': float(object_heading),                # 朝向角度（弧度）
        }
        
        object_info_list.append(object_info)
    
    return object_info_list


def map_command_to_direction(command: np.ndarray) -> str:
    idx = np.argmax(command)
    if idx == 0:
        return "left"
    elif idx == 1:
        return "straight"
    elif idx == 2:
        return "right"
    else:
        return "unknown"
    return "unknown"

def get_history_navigation_infomation(scene) -> Tuple[str, np.ndarray]:
    agent_input = scene.get_agent_input()
    navigation_command = agent_input.ego_statuses
    all_commands = []
    for ego_status in navigation_command:
        cmd = ego_status.driving_command
        all_commands.append(tuple(cmd))
    
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


def get_history_future_trajs(scene:Scene):
    history_traj = scene.get_history_trajectory()
    history_traj = np.array([(pose[0], pose[1]) for pose in history_traj.poses])
    future_traj = scene.get_future_trajectory()
    future_traj = np.array([(pose[0], pose[1]) for pose in future_traj.poses])
    return history_traj, future_traj
    


if __name__=="__main__":

    data_root = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data"
    num_hist_frame = 4
    num_fut_frame = 6
    scene_loader = init_scene_loader(data_root, num_hist_frame, num_fut_frame)


    pass