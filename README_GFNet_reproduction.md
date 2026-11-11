# GFNet (CVPR'25) 复现说明

论文：*Adapting Dense Matching for Homography Estimation with Grid-Based Acceleration* (CVPR 2025)
官方代码：<https://github.com/KN-Zhang/GFNet>（本地基于 commit `2281c4b`，2026-01-15）

## 本机环境

| 项目 | 配置 |
|---|---|
| 代码 | `/home/sy/lgp/GFNet`（上游 clone，不入库） |
| 环境 | conda `lhq-gf`：Python 3.10.13 / torch 2.3.1+cu121 |
| GPU | 2 × RTX 2080 Ti 22G（Turing/SM75），测试用 GPU 0 |
| 数据 | `/home/sy/lgp/ZKNGFNet/output_directory/cvpr25_dataset` |
| 权重 | `/home/sy/lgp/ZKNGFNet/ckpt/{basic,googlemap,vis_ir_drone}/latest.pth` |
| DINOv2 权重 | `TORCH_HOME=/home/sy/lgp/.cache/torch`（自动下载 ViT-L/14） |

## 本地适配（patch 见 `patches/gfnet_local_changes.patch`）

1. **flash-attn 回退**：配置中的 `attention_type` 为 `FLASH2`，但 2080 Ti（SM75）无法使用官方
   FlashAttention2。`CrossFlashAttention2.forward` 在缺少 flash-attn 时改为调用
   `CrossAttention.forward`（相同 softmax 公式，精度等价，仅速度变慢）。
2. **kornia_moons 可选导入**：该包仅用于 `if_print=True` 的可视化，测试流程不使用，改为 try/except。

## 复现步骤

```bash
# 1. 克隆代码并应用本地适配
git clone git@github.com:KN-Zhang/GFNet.git /home/sy/lgp/GFNet
git -C /home/sy/lgp/GFNet apply /home/sy/lgp/patches/gfnet_local_changes.patch

# 2. 链接数据与权重（test.py 要求项目根目录下有 data/ 与 ckpts/）
cd /home/sy/lgp/GFNet
ln -s /home/sy/lgp/ZKNGFNet/output_directory/cvpr25_dataset data
ln -s /home/sy/lgp/ZKNGFNet/ckpt ckpts

# 3. 运行全部测试（5 个数据集，日志写入 /home/sy/lgp/logs/）
bash /home/sy/lgp/scripts/run_gfnet_tests.sh
```

冒烟测试（单对图像，约 1 分钟）：

```bash
TORCH_HOME=/home/sy/lgp/.cache/torch CUDA_VISIBLE_DEVICES=0 \
  /home/sy/anaconda3/envs/lhq-gf/bin/python /home/sy/lgp/scripts/smoke_gfnet.py
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

（待测试完成后更新）
