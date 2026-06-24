import torch
import torch.nn as nn

from diffusion.models.FFP_complex_module import UltimateFrequencyFusionV3
from diffusion.models.swin_transform import SwinTransformer

import torch.nn.functional as F



class DeepDecoder(nn.Module):
    """
    解码 ResNet 深层特征 (2048, H/32, W/32) 生成分割 mask
    """
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

class MultiScaleConditioner(nn.Module):
    def __init__(self, in_channels_list=[128, 256, 512, 1024,1024], 
                 mid_channels=256, 
                 embed_dim=1024, 
                 target_hw=(24, 24)):
        """
        Args:
            mid_channels: 融合后的中间通道数 (e.g., 256)
            embed_dim: LDM 要求的最终 Embedding 维度 (e.g., 1024, 768)
            target_hw: 统一的空间尺寸
        """
        super().__init__()
        self.target_hw = target_hw
        
        # --- Part 1: 多尺度融合 (同前) ---
        total_in = sum(in_channels_list)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_in, mid_channels, kernel_size=1),
            nn.GroupNorm(16, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        )
        
        # --- Part 2: Embedding 投影 (新增加) ---
        # 作用：将 [B, 256, H, W] 映射到 [B, 1024, H, W]
        self.projector = nn.Sequential(
            nn.Conv2d(mid_channels, embed_dim, kernel_size=1), # 1x1卷积升维
            nn.GroupNorm(32, embed_dim),                       # 归一化稳定分布
            nn.SiLU() 
        )
        
        # --- Part 3: 可学习的位置编码 (关键) ---
        # 形状为 [1, 1024, H, W]，并在训练中学习
        self.pos_embedding = nn.Parameter(
            torch.randn(1, embed_dim, target_hw[0], target_hw[1]) * 0.02
        )

    def forward(self, features):
        """
        Returns:
            embedding: [B, H*W, 1024] 
        """
        # 1. 统一尺寸并拼接 (Resize & Concat)
        resized_maps = []
        for x in features:
            if x.shape[-2:] != self.target_hw:
                x = F.interpolate(x, size=self.target_hw, mode='bilinear', align_corners=False)
            resized_maps.append(x)
            
        x = torch.cat(resized_maps, dim=1) # [B, Total_In, 48, 48]
        
        # 2. 融合 (Fusion)
        x = self.fusion_conv(x)            # [B, 256, 48, 48]
        
        # 3. 投影升维 (Projection)
        x = self.projector(x)              # [B, 1024, 48, 48]
        
        # 4. 加上位置编码 (Add Positional Embedding)
        # 广播机制：[B, 1024, 48, 48] + [1, 1024, 48, 48]
        x = x + self.pos_embedding
        
        # 5. 展平与维度置换 (Flatten & Permute)
        # [B, 1024, 48, 48] -> [B, 1024, 2304] -> [B, 2304, 1024]
        B, C, H, W = x.shape
        x = x.flatten(2)       # [B, C, H*W]
        x = x.transpose(1, 2)  # [B, H*W, C]
        
        return x

class SwinBackbone(nn.Module):

    def __init__(self, ckpt_path):
        super().__init__()
        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.t_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
     
        channels = [128, 256, 512, 1024,1024]

        
        self.fusion_layers = nn.ModuleList([
            UltimateFrequencyFusionV3(c) for c in channels
        ])

        self.project = MultiScaleConditioner()
        
        # ---------------------------
        # 3. 加载 22k 预训练权重
        # ---------------------------
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.rgb_swin.load_state_dict(ckpt, strict=False)
        self.t_swin.load_state_dict(ckpt, strict=False)
        self.decoder = DeepDecoder(in_ch=1024, mid_ch=256, out_ch=1)

    def forward(self, x_rgb, x_t):
        ff = []
        fr = self.rgb_swin(x_rgb)
        ft = self.t_swin(x_t)
        
        
        
        for i in range(len(fr)):
            ff.append(self.fusion_layers[i](fr[i], ft[i])[0])
        mid_mask = self.decoder(ff[-1])
        ff = self.project(ff)
        return ff, mid_mask
