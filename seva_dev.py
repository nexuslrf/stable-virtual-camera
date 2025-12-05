import glob
import os
import os.path as osp

import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from seva.model import SGMWrapper
from seva.utils import load_model

import sys
sys.path.append('/home/ruofanl/Projects/video-to-video')
from src.models.custom_unet_mv import UNetMV2DConditionModel

MODEL = SGMWrapper(
    load_model(
        model_version=1.1,
        pretrained_model_name_or_path="stabilityai/stable-virtual-camera",
        weight_name="model.safetensors",
        device="cpu",
        verbose=True,
    ).eval()
) #.to(device)

unet = UNetMV2DConditionModel.from_pretrained('/home/ruofanl/Projects/exp_outputs/SEVA/unet', subfolder="unet")


x = torch.randn(2, 11, 64, 64)
t = torch.ones(1)
d = torch.randn(2, 6, 64, 64)
c = torch.zeros(2, 1, 1024)

with torch.no_grad():
    print('--- SEVA ---')
    y_seva = MODEL.module(x, t, c, d, num_frames=2)
    print('--- UNET ---')
    y_diff = unet(x[None, :], t, c, d[None, :]).sample[0]

    assert torch.allclose(y_seva, y_diff, atol=1e-3)
