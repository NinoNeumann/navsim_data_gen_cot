from navsim.visualization.plots import frame_plot_to_gif, plot_cameras_frame_with_annotations
from navsim.common.dataclasses import Scene
import os
import json
from typing import List


def visualize_cameras_gif(save_path:str, scene:Scene):
    frame_idx = [idx for idx in range(len(scene.frames))]
    num_hist_frame = scene.scene_metadata.num_future_frames
    num_fut_frame = scene.scene_metadata.num_future_frames
    hist_frame_idx = [idx for idx in range(num_hist_frame)]
    fut_frame_idx = [idx for idx in range(num_hist_frame, num_hist_frame + num_fut_frame)]
    
    hist_save_path = os.path.join(save_path, "hist.gif")
    fut_save_path = os.path.join(save_path, "fut.gif")
    full_save_path = os.path.join(save_path, "full.gif")
    frame_plot_to_gif(hist_save_path, plot_cameras_frame_with_annotations, scene, hist_frame_idx)
    frame_plot_to_gif(fut_save_path, plot_cameras_frame_with_annotations, scene, fut_frame_idx)
    frame_plot_to_gif(full_save_path, plot_cameras_frame_with_annotations, scene, frame_idx)

def save_text(save_path:str, text:str):
    with open(save_path, "w") as f:
        f.write(text)

def save_json(save_path:str, data:dict):
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)


