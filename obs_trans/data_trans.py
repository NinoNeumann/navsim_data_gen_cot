import os

os.environ["S3_ENDPOINT"] = "https://obs.cn-southwest-2.huaweicloud.com"
os.environ["S3_USE_HHTPS"] = "0"
os.environ["ACCESS_KEY_ID"] = "HPUAUMBABND5R21BA8CR"
os.environ["SECRET_ACCESS_KEY"] = "GPs3Ag6ahEpm]rEZZmb9bOUlWaCHBVVLYR1rONSV"

import moxing as mox
obs_path = "Partner_Zhu/models"
a = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/mini"
b = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/navsim_logs/mini"
c = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/maps"

d = "/home/ma-user/work/navsim/maps"
e = "/home/ma-user/work/navsim/sensor_blobs/mini"
f = "/home/ma-user/work/navsim/navsim_logs/mini"
mox.file.copy_parallel(a, e)
mox.file.copy_parallel(b, f)
mox.file.copy_parallel(c, d)

