import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 复用之前的先进组件 ----------------
class ResidualUpBlock(nn.Module):
    """使用 Model A 的残差上采样块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        x = self.up(x)
        return self.conv_block(x) + self.conv_skip(x)

class ASPP(nn.Module):
    """使用 Model A 的 ASPP"""
    def __init__(self, in_ch, out_ch, dilations=(1, 6, 12)):
        super().__init__()
        self.modules_list = nn.ModuleList()
        self.modules_list.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.GroupNorm(min(32, out_ch), out_ch), nn.SiLU(inplace=True)))
        for d in dilations:
            self.modules_list.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch), nn.SiLU(inplace=True)))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch), nn.SiLU(inplace=True)
        )
    def forward(self, x):
        res = [m(x) for m in self.modules_list]
        return self.project(torch.cat(res, dim=1))

# ---------------- 融合版主模型 ----------------
class FusionSegHead_Optimized(nn.Module):
    def __init__(self, z_dim=4, f3_dim=1280, base_channels=256, num_classes=1):
        super().__init__()
        
        # 1. 投影层 (SiLU + GroupNorm)
        self.z_proj = nn.Sequential(
            nn.Conv2d(z_dim, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, base_channels),
            nn.SiLU(inplace=True)
        )
        
        # f3 投影：假设 f3 通道数很多，先降维
        self.f3_proj = nn.Sequential(
            nn.Conv2d(f3_dim, base_channels, 1, bias=False),
            nn.GroupNorm(32, base_channels),
            nn.SiLU(inplace=True)
        )

        # 2. 融合后的核心处理：使用 ASPP 替代简单的 Conv
        # 输入通道是 base_channels (如果相加) 或者 2*base_channels (如果Concat)
        # 建议使用 Concat，虽然显存多一点，但特征保留更完整
        self.aspp = ASPP(base_channels * 2, base_channels) 

        # 3. 解码器：使用 ResidualUpBlock
        self.up1 = ResidualUpBlock(base_channels, base_channels // 2)
        self.up2 = ResidualUpBlock(base_channels // 2, base_channels // 4)
        self.up3 = ResidualUpBlock(base_channels // 4, 32) # 最后输出 32 通道

        # 4. Classifier
        self.classifier = nn.Conv2d(32, num_classes, 1)
        
        # 初始化 Output bias，防止训练初期 Loss 爆炸
        nn.init.constant_(self.classifier.bias, -3.0)

    def forward(self, z, f3_dict, thermal_emb=None):
        # 假设 f3 是字典或者是 Tensor
        f3 = f3_dict['mid'] if isinstance(f3_dict, dict) else f3_dict

        # 处理 z
        z_feat = self.z_proj(z) # [B, 256, 60, 80]
        
        # 处理 f3 (对齐到 z 的尺寸)
        f3_feat = F.interpolate(f3, size=z_feat.shape[2:], mode='bilinear', align_corners=False)
        f3_feat = self.f3_proj(f3_feat.float()) # [B, 256, 60, 80]

        # --- 关键改进：Concat 融合 ---
        # 相比 z + f3，Concat 让网络自己决定用 z 的语义还是 f3 的细节
        x = torch.cat([z_feat, f3_feat], dim=1) # [B, 512, 60, 80]

        # 核心特征提取
        x = self.aspp(x) # [B, 256, 60, 80]

        # 上采样
        x = self.up1(x) # -> 120x160
        x = self.up2(x) # -> 240x320
        x = self.up3(x) # -> 480x640

        return self.classifier(x)