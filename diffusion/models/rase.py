import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from diffusion.models.spectral_fusion import UltimateFrequencyFusionV3


import torch


# ------------------ ResNet Splitter (RGB backbone wrapper) ------------------
class ResNet50Splitter(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)

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


class RGBQualityEstimator(nn.Module):
    """Image-level RGB quality correction without normalization layers.

    The input is the original RGB image in [-1, 1].  The zero-initialized
    final layer makes a newly enabled branch initially equivalent to the
    original checkpoint (delta == 0).
    """

    def __init__(self, delta_max=1.0):
        super().__init__()
        self.delta_max = float(delta_max)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, rgb):
        raw = ((rgb.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        raw_delta = self.head(self.features(raw))
        delta = self.delta_max * torch.tanh(raw_delta)
        return delta.view(-1, 1, 1, 1)


# ------------------ ResNet_Fuse_L2 (thermal 单通道 专用, 改进版) ------------------
class ResNet_Fuse_L2(nn.Module):
    def __init__(
        self,
        fuse_channels=256,
        embed_dim=1024,
        pool_size=(8, 8),
        use_phase_loss="train",
        pretrained_backbone=True,
        use_rgb_quality=False,
        rgb_quality_delta_max=1.0,
        disable_deformable_alignment=False,
        stage1_ablation_mode="standard",
    ):
        """
        fuse_channels: 融合时使用的通道数
        embed_dim: tokens 的 embedding 维度
        pool_size: adaptive pool 输出 (ph, pw) -> ph*pw tokens
        """
        super().__init__()

        valid_ablation_modes = {
            "standard",
            "c1_prior_only",
            "c2_rgb_only",
            "c3_rgb_prior",
        }
        if stage1_ablation_mode not in valid_ablation_modes:
            raise ValueError(
                "Unknown stage1_ablation_mode={!r}; expected one of {}".format(
                    stage1_ablation_mode,
                    sorted(valid_ablation_modes),
                )
            )
        # This changes only the feature presented to the direct Stage-I mask
        # decoder. No parameter keys are added, so legacy checkpoints remain
        # strictly loadable in every ablation mode.
        self.stage1_ablation_mode = stage1_ablation_mode

        # ------------------ RGB backbone ------------------
        self.rgb_back = ResNet50Splitter(pretrained=pretrained_backbone)

        # ------------------ Thermal backbone (single-channel) ------------------
        base_th = models.resnet50(weights=None)
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
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 1/4 -> 1/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # 1/2 -> 1
        )

        # ------------------ Fusion 模块（你的自定义频域融合）------------------
        # 注意：fusion channels 要与 projection 输出一致
        # 这里假设 UltimateFrequencyFusionV3 接受 channels=512 并返回 [B,512,H/8,W/8]
        self.fusion = UltimateFrequencyFusionV3(
            channels=512,
            disable_deformable_alignment=disable_deformable_alignment,
        )

        # ------------------ 后续深层 & decoder & tokens head ------------------
        self.decoder = DeepDecoder(in_ch=2048, mid_ch=256, out_ch=1)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.embed_proj = nn.Linear(2048, embed_dim)
        self.pool_size = pool_size

        # Keep the default architecture/state_dict unchanged for legacy
        # checkpoints. The extra module is registered only when explicitly
        # requested by the RGB-quality trainer.
        self.rgb_quality = (
            RGBQualityEstimator(delta_max=rgb_quality_delta_max)
            if use_rgb_quality
            else None
        )

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
        x = self.th_stem(x)  # conv1/bn/relu/maxpool
        x = self.th_layer1(x)  # layer1
        x = self.th_layer2(x)  # layer2

        return x

    def feature_to_tokens(self, feature):
        """Convert a 512-channel early feature to the standard OSD tokens."""
        deep_l3 = self.rgb_back.forward_layer3(feature)
        deep_l4 = self.rgb_back.forward_layer4(deep_l3)
        pooled = self.pool(deep_l4)
        batch, channels, ph, pw = pooled.shape
        tokens = pooled.permute(0, 2, 3, 1).reshape(batch, ph * pw, channels)
        return self.embed_proj(tokens)

    def forward(self, x_rgb, x_th, return_diagnostics=False):
        """Return prior tokens, reliability summaries, auxiliary logits,
        phase/spatial losses, and the fused shallow feature.

        RGB has shape [B,3,H,W]; thermal has shape [B,1,H,W].
        Optional diagnostics contain the underlying reliability tensors.
        """
        if x_th.ndim != 4 or x_th.shape[1] != 1:
            raise ValueError(
                "Thermal input must be shape [B,1,H,W]. Got: {}".format(x_th.shape)
            )

        rgb_quality_delta = (
            self.rgb_quality(x_rgb) if self.rgb_quality is not None else None
        )

        # ---------- Backbone early features ----------
        fr2 = self.rgb_back.forward_early(x_rgb)  # [B,512,H/8,W/8]
        # C2 is a true RGB-only direct-decoding control. Skip the Thermal and
        # R-ASE paths so their activations and BatchNorm state cannot affect it.
        if self.stage1_ablation_mode == "c2_rgb_only":
            fused_l2 = fr2
            decoder_l2 = fr2
            zero = fr2.new_zeros(())
            align_loss = zero
            loss_align_spatial = zero
            monitor = {
                "w_rgb_mean": 0.0,
                "w_th_mean": 0.0,
                "gate_mean": 0.0,
                "res_gate": 0.0,
                "align_scale": 0.0,
                "align_scale_effective": 0.0,
                "u_rgb_base_mean": 0.0,
                "u_rgb_pre_mean": 0.0,
                "u_th_pre_mean": 0.0,
                "u_th_post_mean": 0.0,
                "rgb_quality_delta_mean": 0.0,
                "rgb_ratio_mean": 1.0,
                "var_rgb_mean": 0.0,
                "var_th_mean": 0.0,
            }
            fusion_diagnostics = {}
            thermal_feature_pre_scale = None
        else:
            ft2 = self.forward_th_early(x_th)  # [B,512,H/8,W/8]
            thermal_feature_pre_scale = ft2 if return_diagnostics else None

            # ---------- Scale-match pt -> pr (batch-wise) ----------
            eps = 1e-6
            pr_std = (
                fr2.view(fr2.shape[0], -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1)
            )
            pt_std = (
                ft2.view(ft2.shape[0], -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1)
            )
            scale = (pr_std + eps) / (pt_std + eps)
            ft2 = ft2 * scale

            # ---------- Fusion (RGB, Thermal) ----------
            # Complex FFT and phase operations need FP32 even if the surrounding
            # ResNet is run with CUDA autocast.
            with torch.autocast(device_type=fr2.device.type, enabled=False):
                (
                    fused_l2,
                    align_loss,
                    monitor,
                    fusion_diagnostics,
                    loss_align_spatial,
                ) = self.fusion(
                    fr2.float(),
                    ft2.float(),
                    rgb_quality_delta=(
                        rgb_quality_delta.float()
                        if rgb_quality_delta is not None
                        else None
                    ),
                    return_diagnostics=return_diagnostics,
                )

            if self.stage1_ablation_mode == "c3_rgb_prior":
                # Parameter-free fusion keeps the downstream architecture,
                # parameter count and optimization budget identical.
                decoder_l2 = 0.5 * (fr2.float() + fused_l2)
            else:
                # The standard path and C1 both decode the R-ASE prior alone.
                decoder_l2 = fused_l2

        deep_l3 = self.rgb_back.forward_layer3(decoder_l2)  # [B,1024,H/16,W/16]
        deep_l4 = self.rgb_back.forward_layer4(deep_l3)  # [B,2048,H/32,W/32]

        # ---------- Decoder ----------
        mid_mask = self.decoder(deep_l4)  # [B,1,H,W]

        # ---------- Tokens ----------
        pooled = self.pool(deep_l4)  # [B,2048,ph,pw]
        B, C, ph, pw = pooled.shape
        tokens = pooled.permute(0, 2, 3, 1).reshape(B, ph * pw, C)  # [B,ph*pw,2048]
        tokens = self.embed_proj(tokens)  # [B,ph*pw,embed_dim]

        outputs = (tokens, monitor, mid_mask, align_loss, loss_align_spatial, fused_l2)
        if return_diagnostics:
            fusion_diagnostics["thermal_feature_pre_scale"] = thermal_feature_pre_scale
            fusion_diagnostics["rgb_quality_delta"] = (
                rgb_quality_delta
                if rgb_quality_delta is not None
                else torch.zeros(
                    x_rgb.shape[0],
                    1,
                    1,
                    1,
                    device=x_rgb.device,
                    dtype=x_rgb.dtype,
                )
            )
            return (*outputs, fusion_diagnostics)
        return outputs
