import os
import moxing as mox

paths = [
    "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/mini/2021.05.12.22.00.38_veh-35_01008_01518/CAM_F0/ed564f55883d50d3.jpg",
    "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/mini/2021.05.12.22.00.38_veh-35_01008_01518/CAM_L0/8c30cd489a2d51e4.jpg",
    "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/mini/2021.05.12.22.00.38_veh-35_01008_01518/CAM_R0/4db22fb527ea5934.jpg",
    "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/mini/2021.05.12.22.00.38_veh-35_01008_01518/CAM_L1/ec2f6153d8585b6f.jpg",
    "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/mini/2021.05.12.22.00.38_veh-35_01008_01518/CAM_R1/0ab83122318756e9.jpg",
    "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/mini/2021.05.12.22.00.38_veh-35_01008_01518/CAM_B0/867ddcf80e585c75.jpg",
    "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/mini/2021.05.12.22.00.38_veh-35_01008_01518/CAM_L2/aa37d52386fe5bf5.jpg",
    "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/mini/2021.05.12.22.00.38_veh-35_01008_01518/CAM_R2/a4df8965b9515143.jpg"
]

save_root = "./tmp_pics"

def trans(paths, save_root):
    os.makedirs(save_root, exist_ok=True)
    local_paths = []
    for obs_path in paths:
        filename = os.path.basename(obs_path)           # 原始文件名
        cam_dir = os.path.basename(os.path.dirname(obs_path))  # 提取相机目录 (CAM_F0, CAM_L0 ...)
        # 拼接相机信息到文件名
        new_filename = f"{cam_dir}_{filename}"
        local_path = os.path.join(save_root, new_filename)
        
        # 拷贝
        mox.file.copy(obs_path, local_path)
        print(f"Copied: {obs_path} -> {local_path}")
        local_paths.append(local_path)
    return local_paths

# 使用示例
local_files = trans(paths, save_root)
print("本地文件：", local_files)
