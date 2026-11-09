import os
from huggingface_hub import snapshot_download

# 1. 设置镜像（加速下载）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 执行下载
print("开始下载...")
snapshot_download(
    repo_id="carney1212/cvpr25_datasets",
    repo_type="dataset",
    local_dir="/home/sy/lhq/data/ZKNGFNet",  # <--- 你的目标路径
    local_dir_use_symlinks=False,             # 下载真实文件，不要软链接
    resume_download=True                      # 支持断点续传
)
print("下载完成！")
