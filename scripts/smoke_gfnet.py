"""快速冒烟测试：加载 googlemap 权重，对 1 对图像跑完整估计流程。"""
import os
import sys
import json

os.chdir('/home/sy/lgp/GFNet')
sys.path.insert(0, '/home/sy/lgp/GFNet')

import torch
from model.network import GFNet
from estimation import demo_estimation

conf = json.load(open('gfnet_configs/map.json'))
model = GFNet(conf=conf, initial_res=(448, 448), upsample_res=(560, 560),
              symmetric=True, upsample_preds=True, attenuate_cert=True).cuda()
states = torch.load('ckpts/googlemap/latest.pth')
model.load_state_dict(states['model'])

base = 'data/test/googlemap_1k_448x448_new'
img = sorted(os.listdir(f'{base}/source'))[0]
img1 = f'{base}/source/{img}'
img2 = f'{base}/target/{img}'
H = f'{base}/H_s2t/{os.path.splitext(img)[0]}.json'
err, t = demo_estimation(model, img1, img2, H)
print(f'smoke OK: ACE={err:.4f}, time={t:.3f}s')
