import os

os.environ["S3_ENDPOINT"] = "https://obs.cn-southwest-2.huaweicloud.com"
os.environ["S3_USE_HHTPS"] = "0"
os.environ["ACCESS_KEY_ID"] = "HPUAUMBABND5R21BA8CR"
os.environ["SECRET_ACCESS_KEY"] = "GPs3Ag6ahEpm]rEZZmb9bOUlWaCHBVVLYR1rONSV"

import moxing as mox
import icecream as ic
obs_path = "obs://yw-2030-extern/Partner_Zhu/models/Qwen2.5-VL-32B-Instruct"
model_path = "/home/ma-user/work/Qwen2.5-VL-32B-Instruct"
mox.file.copy_parallel(model_path, obs_path)