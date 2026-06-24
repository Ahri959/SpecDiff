import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 轻量级基础组件 (MobileNet风格) ----------------
class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积：将 3x3 卷积拆分为 Depthwise + Pointwise。
    大幅减少参数量和 FLOPs，非常适合轻量级 Head。
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, 
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(min(16, out_ch), out_ch) # GN 比 BN 在小 Batch 下更稳定
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class LightUpBlock(nn.Module):
    """
    轻量级上采样模块
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # 既然已经上采样了，为了轻量，这里只用一个 DWConv 整理特征
        self.conv = DepthwiseSeparableConv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

# ---------------- 最终的轻量级 Head ----------------
class LightSegHead(nn.Module):
    def __init__(self, z_dim=4, f3_dim=1280, dim=128, num_classes=1):
        """
        Args:
            z_dim: LDM latent channels (usually 4)
            f3_dim: UNet mid-block channels (usually 1280)
            dim: 内部特征维度，越小越快 (建议 64, 96, 或 128)
        """
        super().__init__()
        
        # 1. 语义特征适配 (最重的一步，立刻降维)
        # 1280 -> dim
        self.sem_proj = nn.Sequential(
            nn.Conv2d(f3_dim, dim, 1, bias=False), # 1x1 Conv 降维
            nn.GroupNorm(min(16, dim), dim),
            nn.SiLU(inplace=True)
        )

        # 2. 空间特征适配 (z)
        # 4 -> dim
        self.spat_proj = nn.Sequential(
            nn.Conv2d(z_dim, dim, 1, bias=False),
            nn.GroupNorm(min(16, dim), dim),
            nn.SiLU(inplace=True)
        )

        # 3. 融合层 (在 Latent 分辨率下做)
        # 输入是 concat(sem, spat) -> 2*dim -> dim
        self.fusion = DepthwiseSeparableConv(dim * 2, dim)

        # 4. Decoder (从 H/8 恢复到 H)
        # 假设 dim=128
        # Block1: H/8 -> H/4
        self.up1 = LightUpBlock(dim, dim // 2)      # 128 -> 64
        # Block2: H/4 -> H/2
        self.up2 = LightUpBlock(dim // 2, dim // 4) # 64 -> 32
        # Block3: H/2 -> H
        self.up3 = LightUpBlock(dim // 4, 16)       # 32 -> 16

        # 5. Classifier
        self.classifier = nn.Conv2d(16, num_classes, 1)
        
        # 初始化偏置，防止训练初期 Loss 爆炸
        nn.init.constant_(self.classifier.bias, -2.0)

    def forward(self, z, f3_dict, thermal_emb=None):
        """
        z: [B, 4, H/8, W/8]
        f3: [B, 1280, H/64, W/64] (SD的典型中层尺寸)
        """
        f3 = f3_dict['mid'] if isinstance(f3_dict, dict) else f3_dict

        # 1. 处理语义特征 (Semantic)
        sem = self.sem_proj(f3) # [B, 1280, h, w] -> [B, dim, h, w]
        # 上采样到 z 的尺寸
        sem = F.interpolate(sem, size=z.shape[2:], mode='bilinear', align_corners=False)

        # 2. 处理空间特征 (Spatial / Latent)
        spat = self.spat_proj(z) # [B, dim, H/8, W/8]

        # 3. 简单拼接融合
        # 此时 sem 提供“是什么”，spat 提供“在哪里”
        x = torch.cat([sem, spat], dim=1) # [B, 2*dim, H/8, W/8]
        x = self.fusion(x)                # [B, dim, H/8, W/8]

        # 4. 逐级上采样解码
        x = self.up1(x) # -> H/4
        x = self.up2(x) # -> H/2
        x = self.up3(x) # -> H

        # 5. 输出
        out = self.classifier(x)
        
        return out