import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================
#  RobustSaliencyFocus
# ================================
class RobustSaliencyFocus(nn.Module):
    def __init__(self, in_channels=2048, groups=32):
        super().__init__()
        
        # Attention Head
        self.focus_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=in_channels // 2),
            nn.SiLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )
        
        # Feature Refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            nn.SiLU()
        )
        
        # Zero-Init Scale
        self.scale = nn.Parameter(torch.zeros(1))

 
        self.proj = nn.Linear(2048, 1024)
        # 使用 adaptive pool 调整 token 数量
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # 8x8=64 token


    def get_soft_label(self, gt_mask, target_size):
        """下采样 + 平滑生成软标签"""
        gt_small = F.interpolate(gt_mask, size=target_size, mode='bilinear', align_corners=False)
        gt_soft = F.avg_pool2d(gt_small, kernel_size=3, stride=1, padding=1)
        return gt_soft

    def hybrid_loss(self, pred_logits, gt_soft):
        """Focal Loss + Dice Loss"""
        pred_sigmoid = torch.sigmoid(pred_logits)
        
        # --- Focal Loss ---
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_soft, reduction='none')
        p_t = pred_sigmoid * gt_soft + (1 - pred_sigmoid) * (1 - gt_soft)
        gamma = 2.0
        focal_factor = (1.0 - p_t) ** gamma
        focal_loss = (focal_factor * bce_loss).mean()
        
        # --- Dice Loss ---
        intersection = (pred_sigmoid * gt_soft).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + gt_soft.sum(dim=(2, 3))
        dice = 1 - (2. * intersection + 1) / (union + 1)
        
        return focal_loss + dice.mean()

    def forward(self, thermal_feat, gt_mask=None):
        # Attention map
        attn_logits = self.focus_head(thermal_feat)
        attn_map = torch.sigmoid(attn_logits)
        
        # Soft fusion with scale
        out_feat = thermal_feat * (1 + self.scale * attn_map)
        out_feat = self.refine_conv(out_feat)
        
        aux_loss = torch.tensor(0., device=thermal_feat.device)
        if self.training and gt_mask is not None:
            gt_soft = self.get_soft_label(gt_mask, attn_logits.shape[-2:])
            aux_loss = self.hybrid_loss(attn_logits, gt_soft)
            
        out_feat_con = self.pool(out_feat)   # [B, 2048, 8, 8]

        B, C, H, W = out_feat_con.shape
        out_feat_con = out_feat_con.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, 64, 2048]
        out_feat_con = self.proj(out_feat_con)  # 映射到 embed_dim
            
        return out_feat,out_feat_con, aux_loss, attn_map


# ================================
#  SynergisticRectifier
# ================================
class SynergisticRectifier(nn.Module):
    def __init__(self, thermal_channels=2048, latent_channels=4, bias_hidden=256):
        super().__init__()
        self.align_conv = nn.Sequential(
            nn.Conv2d(thermal_channels, bias_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(bias_hidden, latent_channels, kernel_size=1)
        )
        
        self.output_layer = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        for p in self.output_layer.parameters():
            if p.requires_grad:
                nn.init.zeros_(p)

        self.bias_scale = nn.Parameter(torch.tensor(0.1))  # 控制初始幅度

        self.chan_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(thermal_channels, thermal_channels // 8, 1),
            nn.SiLU(),
            nn.Conv2d(thermal_channels // 8, latent_channels, 1),
            nn.Sigmoid()
        )

        self.global_confine = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(thermal_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z_rgb, thermal_feat, sfc_attn_map=None):
        # Align spatially
        if thermal_feat.shape[-2:] != z_rgb.shape[-2:]:
            thermal_resized = F.interpolate(thermal_feat, size=z_rgb.shape[-2:], mode='bilinear', align_corners=False)
        else:
            thermal_resized = thermal_feat

        # Predict bias
        rectify_bias = self.align_conv(thermal_resized)
        # ✅ 删除手动归一化
        # rectify_bias = rectify_bias / (rectify_bias.abs().mean(dim=[2,3], keepdim=True) + 1e-6)

        # Channel gating
        chan_context = F.adaptive_avg_pool2d(thermal_feat, 1)
        ch_gate = self.chan_gate(chan_context)
        
        # Global confidence
        global_conf = self.global_confine(thermal_feat).view(-1,1,1,1)
        
        # Spatial confidence
        if sfc_attn_map is not None:
            if sfc_attn_map.shape[-2:] != rectify_bias.shape[-2:]:
                sfc_attn_resized = F.interpolate(sfc_attn_map, size=rectify_bias.shape[-2:], mode='bilinear', align_corners=False)
            else:
                sfc_attn_resized = sfc_attn_map
        else:
            sfc_attn_resized = torch.ones(rectify_bias.shape[0], 1, rectify_bias.shape[2], rectify_bias.shape[3], device=rectify_bias.device)

        # Assemble bias
        ch_gate = ch_gate.expand(-1, -1, rectify_bias.shape[2], rectify_bias.shape[3])
        spatial_conf = sfc_attn_resized.expand(-1, rectify_bias.shape[1], -1, -1)
        bias = self.bias_scale * rectify_bias * ch_gate * spatial_conf * global_conf

        shift = self.output_layer(bias)
        z_rectified = z_rgb + shift

        return z_rectified, {'global_conf': global_conf.detach(), 'chan_gate': ch_gate.detach()}
