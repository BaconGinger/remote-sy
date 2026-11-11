#!/usr/bin/env bash
# GFNet (CVPR'25) 复现测试脚本
# 在 2×RTX 2080 Ti 环境下逐个数据集运行官方评测流程
set -euo pipefail

cd /home/sy/lgp/GFNet/code
export TORCH_HOME=/home/sy/lgp/GFNet/.cache/torch
export PYTHONPATH=/home/sy/lgp/GFNet/code/third_party/romatch_pkg${PYTHONPATH:+:$PYTHONPATH}
# 当前机器仅剩一块可用 GPU（重启后枚举为 0）；可通过 GPU=... 覆盖
export CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=/home/sy/anaconda3/envs/lhq-gf/bin/python
LOGDIR=/home/sy/lgp/GFNet/logs
mkdir -p "$LOGDIR"

run() {
    local ds=$1 conf=$2 ckpt=$3
    echo "===== $(date '+%F %T') START $ds =====" | tee -a "$LOGDIR/summary.log"
    "$PY" -m test --dataset "$ds" --conf_path "$conf" --ckpt_path "$ckpt" \
        2>&1 | tee "$LOGDIR/${ds}.log"
    echo "===== $(date '+%F %T') END   $ds =====" | tee -a "$LOGDIR/summary.log"
}

run googlemap_448x448 gfnet_configs/map.json    ckpts/googlemap/latest.pth
run googlemap_224x224 gfnet_configs/map.json    ckpts/googlemap/latest.pth
run googlemap_672x672 gfnet_configs/map.json    ckpts/googlemap/latest.pth
run mscoco            gfnet_configs/basic.json  ckpts/basic/latest.pth
run vis_ir_drone      gfnet_configs/vis_ir.json ckpts/vis_ir_drone/latest.pth
