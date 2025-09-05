import os

os.environ["S3_ENDPOINT"] = "https://obs.cn-southwest-2.huaweicloud.com"
os.environ["S3_USE_HHTPS"] = "0"
os.environ["ACCESS_KEY_ID"] = "HPUAUMBABND5R21BA8CR"
os.environ["SECRET_ACCESS_KEY"] = "GPs3Ag6ahEpm]rEZZmb9bOUlWaCHBVVLYR1rONSV"

import moxing as mox
obs_path = "Partner_Zhu/models"
a = "obs://yw-2030-extern/Partner_Zhu/code/navsim_cot_gen/navsim/maps"
b = "/home/ma-user/work/navsim/maps"
mox.file.copy_parallel(b,a)
mox.file.remove

c = "obs://yw-2030-extern/Partner_Zhu/code/navsim_cot_gen/code"
d = "/home/ma-user/work/navsim_data_gen_cot"
mox.file.copy_parallel(d,c)


