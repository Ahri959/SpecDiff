import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from diffusion.models.FFP_complex_module import UltimateFrequencyFusionV3

import matplotlib.pyplot as plt


import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from diffusion.models.visual_fusion import run_paper_visualizations

def save_resnet_structure_map(
    feature_map_tensor,
    filename="heatmap.png",
    percentile=5,
    colormap=cv2.COLORMAP_TURBO,
):
    if feature_map_tensor.ndim == 4:
        fmap = feature_map_tensor[0]
    else:
        fmap = feature_map_tensor

    fmap = fmap.detach().float().cpu()  # [C, H, W]

    # ★ ResNet 特征最稳的方式
    heatmap = fmap.var(dim=0).numpy()

    lo, hi = np.percentile(heatmap, [percentile, 100 - percentile])
    heatmap = np.clip(heatmap, lo, hi)
    heatmap = (heatmap - lo) / (hi - lo + 1e-6)

    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, colormap)
    cv2.imwrite(filename, heatmap_color)





# ------------------ ResNet Splitter (RGB backbone wrapper) ------------------
class ResNet50Splitter(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)

        # early
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2

        # deep
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward_early(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        return f2  # [B,512,H/8,W/8]

    def forward_layer3(self, x):
        return self.layer3(x)  # 输入 [B,512,...] -> 输出 [B,1024,...]

    def forward_layer4(self, x):
        return self.layer4(x)  # 输入 [B,1024,...] -> 输出 [B,2048,...]


# ------------------ DeepDecoder ------------------
class DeepDecoder(nn.Module):
    def __init__(self, in_ch=2048, mid_ch=256, out_ch=1):
        super().__init__()
        # Stage1: 1/32 -> 1/16
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch * 4, mid_ch * 2, kernel_size=2, stride=2),
        )
        # Stage2: 1/16 -> 1/8
        self.up2 = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch * 2, mid_ch, kernel_size=2, stride=2),
        )
        # Stage3: 1/8 -> 1/4
        self.up3 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, mid_ch // 2, kernel_size=2, stride=2),
        )
        # Stage4: 1/4 -> 1/2
        self.up4 = nn.Sequential(
            nn.Conv2d(mid_ch // 2, mid_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch // 2, mid_ch // 4, kernel_size=2, stride=2),
        )
        # Stage5: 1/2 -> 1
        self.up5 = nn.Sequential(
            nn.Conv2d(mid_ch // 4, mid_ch // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch // 4, out_ch, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x  # [B,1,H,W]


# ------------------ ResNet_Fuse_L2 (thermal 单通道 专用, 改进版) ------------------
class ResNet_Fuse_L2(nn.Module):
    def __init__(self, fuse_channels=256, embed_dim=1024,
                 pool_size=(8, 8), use_phase_loss='train',
                 pretrained_backbone=True):
        """
        fuse_channels: 融合时使用的通道数
        embed_dim: tokens 的 embedding 维度
        pool_size: adaptive pool 输出 (ph, pw) -> ph*pw tokens
        """
        super().__init__()

        # ------------------ RGB backbone ------------------
        self.rgb_back = ResNet50Splitter(pretrained=pretrained_backbone)

        # ------------------ Thermal backbone (single-channel) ------------------
        base_th = models.resnet50(pretrained=False)
        base_th.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # thermal early stem + layer1/layer2
        self.th_stem = nn.Sequential(
            base_th.conv1, base_th.bn1, base_th.relu, base_th.maxpool
        )
        self.th_layer1 = base_th.layer1
        self.th_layer2 = base_th.layer2

        # ------------------ thermal pre-enhance block (小卷积增强对比度) ------------------
        # 目的是在进入 backbone 前提升 thermal 的局部对比度/纹理信息
        self.th_pre = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
            # don't put strong normalization here; keep it learnable
        )

        # ------------------ thermal shallow decoder (deep supervision head) ------------------
        # 从 ft2 ([B,512,H/8,W/8]) 解码到原图分辨率 [B,1,H,W]
        self.th_decoder = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 1/8 -> 1/4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 1/4 -> 1/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),    # 1/2 -> 1
        )

        # ------------------ Fusion 模块（你的自定义频域融合）------------------
        # 注意：fusion channels 要与 projection 输出一致
        # 这里假设 UltimateFrequencyFusionV3 接受 channels=512 并返回 [B,512,H/8,W/8]
        self.fusion = UltimateFrequencyFusionV3(
            channels=512,
        )

        # ------------------ 后续深层 & decoder & tokens head ------------------
        self.decoder = DeepDecoder(in_ch=2048, mid_ch=256, out_ch=1)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.embed_proj = nn.Linear(2048, embed_dim)
        self.pool_size = pool_size

    def forward_th_early(self, x_th):
        """
        thermal 单通道 early forward: pre-enhance -> stem -> layer1 -> layer2
        输入 x_th: [B,1,H,W]
        输出: ft2: [B,512,H/8,W/8]
        """
        # 1) per-sample spatial normalization (keep contrast but standardize scale)
        b, c, h, w = x_th.shape
        mean = x_th.view(b, c, -1).mean(dim=-1).view(b, c, 1, 1)
        std = x_th.view(b, c, -1).std(dim=-1).view(b, c, 1, 1) + 1e-6
        x = (x_th - mean) / std

        # 2) small learnable pre-enhance to boost local contrast/texture
        x = self.th_pre(x)  # [B,1,H,W]

        # 3) stem & layers
        x = self.th_stem(x)     # conv1/bn/relu/maxpool
        x = self.th_layer1(x)   # layer1
        x = self.th_layer2(x)   # layer2

        return x

    def forward(self, x_rgb, x_th, debug=False, debug_dir=None):
        """
        x_rgb: [B,3,H,W]
        x_th:  [B,1,H,W]  <- 强制要求单通道
        debug: 如果为 True，会输出若干热图到 debug_dir（会创建该目录）
        return: tokens, monitor, mid_mask, align_loss, mask_th
        """
        if x_th.ndim != 4 or x_th.shape[1] != 1:
            raise ValueError("Thermal input must be shape [B,1,H,W]. Got: {}".format(x_th.shape))

        if debug and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)

        # ---------- Backbone early features ----------
        fr2 = self.rgb_back.forward_early(x_rgb)   # [B,512,H/8,W/8]
        ft2 = self.forward_th_early(x_th)          # [B,512,H/8,W/8]

    

        # ---------- Scale-match pt -> pr (batch-wise) ----------
        eps = 1e-6
        pr_std = fr2.view(fr2.shape[0], -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1)  # [B,1,1,1]
        pt_std = ft2.view(ft2.shape[0], -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1)  # [B,1,1,1]
        scale = (pr_std + eps) / (pt_std + eps)
        ft2 = ft2 * scale
        
      

        # ---------- Fusion (RGB, Thermal) ----------
        # 注意：fusion 的签名和返回（fused, align_loss, monitor）依赖于你的实现
        fused_l2, align_loss, monitor,debug,loss_align_spatial= self.fusion(fr2, ft2)  # fused_small: [B, 512, H/8, W/8]
        
        #run_paper_visualizations(debug)
        
    

        deep_l3 = self.rgb_back.forward_layer3(fused_l2)   # [B,1024,H/16,W/16]
        deep_l4 = self.rgb_back.forward_layer4(deep_l3)    # [B,2048,H/32,W/32]

        # ---------- Decoder ----------
        mid_mask = self.decoder(deep_l4)  # [B,1,H,W]


        # ---------- Tokens ----------
        pooled = self.pool(deep_l4)  # [B,2048,ph,pw]
        B, C, ph, pw = pooled.shape
        tokens = pooled.permute(0, 2, 3, 1).reshape(B, ph * pw, C)  # [B,ph*pw,2048]
        tokens = self.embed_proj(tokens)  # [B,ph*pw,embed_dim]

        return tokens, monitor, mid_mask, align_loss,loss_align_spatial , fused_l2


# ------------------ 简单 sanity check（可运行） ------------------
if __name__ == "__main__":
    model = ResNet_Fuse_L2(pretrained_backbone=True)
    model.eval()
    B, H, W = 2, 384, 384
    x_rgb = torch.randn(B, 3, H, W)
    x_th = torch.randn(B, 1, H, W)

    with torch.no_grad():
        tokens, monitor, mid_mask, align_loss, mask_th = model(x_rgb, x_th, debug=True, debug_dir="./debug_vis_test")

    print("tokens:", tokens.shape)            # [B,ph*pw,embed_dim]
    print("mid_mask:", mid_mask.shape)        # [B,1,H,W]
    print("mask_th:", mask_th.shape)          # [B,1,H,W]
    print("align_loss:", align_loss)
    if isinstance(monitor, dict):
        print("monitor keys:", list(monitor.keys()))
    else:
        print("monitor:", monitor)

    # usage example for training:
    # loss_main = BCEWithLogitsLoss(mid_mask, GT_main)
    # loss_th   = BCEWithLogitsLoss(mask_th, GT_th)
    # total_loss = loss_main + 0.4 * loss_th  # deep supervision 权重建议 0.3~0.5
