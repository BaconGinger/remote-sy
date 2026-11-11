# GFNet (CVPR'25) 复现说明

论文：*Adapting Dense Matching for Homography Estimation with Grid-Based Acceleration* (CVPR 2025)
官方代码：<https://github.com/KN-Zhang/GFNet>（本地基于 commit `2281c4b`，2026-01-15）

## 目录结构

```
/home/sy/lgp/GFNet/               # 实验根目录
├── code/                         # 官方代码 clone（保留上游 git）
│   ├── data -> /home/sy/lgp/ZKNGFNet/output_directory/cvpr25_dataset
│   └── ckpts -> /home/sy/lgp/ZKNGFNet/ckpt
├── scripts/                      # 批量测试、冒烟测试
├── patches/                      # 本地适配补丁
├── paper/GFNet_CVPR2025.pdf      # 论文（不入库）
├── logs/                         # 运行日志（不入库）
├── results/                      # 汇总结果
├── .cache/torch/                 # DINOv2 权重缓存（不入库）
└── README.md
```

## 本机环境

| 项目 | 配置 |
|---|---|
| 环境 | conda `lhq-gf`：Python 3.10.13 / torch 2.3.1+cu121 |
| GPU | 2 × RTX 2080 Ti 22G（Turing/SM75） |
| 数据 | `/home/sy/lgp/ZKNGFNet/output_directory/cvpr25_dataset` |
| 权重 | `/home/sy/lgp/ZKNGFNet/ckpt/{basic,googlemap,vis_ir_drone}/latest.pth` |
| DINOv2 权重 | `TORCH_HOME=/home/sy/lgp/GFNet/.cache/torch`（ViT-L/14，约 1.2G） |

## 本地适配（patch 见 `patches/gfnet_local_changes.patch`）

1. **flash-attn 回退**：配置中的 `attention_type` 为 `FLASH2`，但官方 FlashAttention2
   不支持 2080 Ti（SM75）。`CrossFlashAttention2.forward` 在缺少 flash-attn 时改用
   `F.scaled_dot_product_attention`：数学上同样是 softmax 注意力，但和 flash-attn 一样
   使用 fp32 softmax 累加，避免 fp16 logit 溢出产生 inf/NaN。
2. **kornia_moons 可选导入**：该包仅用于 `if_print=True` 的可视化，测试流程不使用，改为 try/except。

## 复现步骤

```bash
# 1. 克隆代码并应用本地适配
git clone git@github.com:KN-Zhang/GFNet.git /home/sy/lgp/GFNet/code
git -C /home/sy/lgp/GFNet/code apply /home/sy/lgp/GFNet/patches/gfnet_local_changes.patch

# 2. 安装 romatch（RoMa 工具包，仅一个导入依赖，装到 third_party 避免改动共享环境）
cd /home/sy/lgp/GFNet/code
pip install --no-deps --target third_party/romatch_pkg romatch

# 3. 链接数据与权重（test.py 要求项目根目录下有 data/ 与 ckpts/）
ln -s /home/sy/lgp/ZKNGFNet/output_directory/cvpr25_dataset data
ln -s /home/sy/lgp/ZKNGFNet/ckpt ckpts

# 4. 运行全部测试（5 个数据集，日志写入 /home/sy/lgp/GFNet/logs/）
bash /home/sy/lgp/GFNet/scripts/run_gfnet_tests.sh
```

脚本内环境变量：`TORCH_HOME=/home/sy/lgp/GFNet/.cache/torch`、
`PYTHONPATH=/home/sy/lgp/GFNet/code/third_party/romatch_pkg`、
`CUDA_VISIBLE_DEVICES=${GPU:-1}`（可用 `GPU=0` 覆盖）。

冒烟测试（单对图像，约 1 分钟）：

```bash
cd /home/sy/lgp/GFNet/code
CUDA_VISIBLE_DEVICES=1 \
/home/sy/anaconda3/envs/lhq-gf/bin/python /home/sy/lgp/GFNet/scripts/smoke_gfnet.py
```

## 评测协议与预期指标

`test.py` 流程：448 输入 / 560 上采样 → 采样 5000 匹配 → `cv2.findHomography`
(RANSAC, 3px) → 四角平均误差 ACE（上限 70px）→ AUC@{3,5,10,20}。

论文 Table 1（448×448，Group B 图像匹配组）的 GFNet 数值，作为对照：

| 数据集 | auc@3 | auc@5 | auc@10 | auc@20 |
|---|---|---|---|---|
| MSCOCO | 97.67 | 98.61 | 99.3 | 99.65 |
| VIS-IR | 21.28 | 35.72 | 58.44 | 72.33 |
| GoogleMap | 51.44 | 66.34 | 80.62 | 89.66 |

googlemap_224x224 / 672x672 无表格数值，论文以 Figure 4 曲线给出，主要看趋势
（分辨率下降 AUC 降低、672 基本持平）。

## 复现结果

- 2026-11-11 首次运行：`googlemap_448x448` 跑到 536/1000 时 GPU 0 掉卡
  （`NVRM: Xid 79, GPU has fallen off the bus`），CUDA 全局不可用，运行中断。
  该次 NaN 报错很可能是掉卡前的计算损坏所致，已同时加入 SDPA 数值稳定回退。
- 待机器重启、GPU 恢复后重新运行并在此更新指标。
