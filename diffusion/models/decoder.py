import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Reuse the Advanced Components Defined Earlier ----------------
class ResidualUpBlock(nn.Module):
    """Residual upsampling block from Model A."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )
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
    """Atrous Spatial Pyramid Pooling module from Model A."""

    def __init__(self, in_ch, out_ch, dilations=(1, 6, 12)):
        super().__init__()
        self.modules_list = nn.ModuleList()
        self.modules_list.append(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
        )
        for d in dilations:
            self.modules_list.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                    nn.GroupNorm(min(32, out_ch), out_ch),
                    nn.SiLU(inplace=True),
                )
            )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        res = [m(x) for m in self.modules_list]
        return self.project(torch.cat(res, dim=1))


class MultiScaleResidualRefinement(nn.Module):
    """Use the two highest-resolution UNet features for residual refinement.

    The final convolution is zero-initialized, so enabling this branch leaves
    an old checkpoint's step-zero prediction unchanged.
    """

    def __init__(self, mid_channels=640, high_channels=320, out_channels=256):
        super().__init__()
        refine_channels = out_channels // 2
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_channels, refine_channels, 1, bias=False),
            nn.GroupNorm(32, refine_channels),
            nn.SiLU(inplace=True),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, refine_channels, 1, bias=False),
            nn.GroupNorm(32, refine_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(refine_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, mid_feature, high_feature, output_size, output_dtype):
        mid = self.mid_proj(mid_feature)
        high = self.high_proj(high_feature)
        mid = F.interpolate(
            mid.float(), size=output_size, mode="bilinear", align_corners=False
        )
        high = F.interpolate(
            high.float(), size=output_size, mode="bilinear", align_corners=False
        )
        if mid.dtype != output_dtype:
            mid = mid.to(output_dtype)
            high = high.to(output_dtype)
        return self.fuse(torch.cat([mid, high], dim=1))


# ---------------- Main Fusion Model ----------------
class FusionSegHead_Optimized(nn.Module):
    def __init__(
        self,
        z_dim=4,
        f3_dim=1280,
        base_channels=256,
        num_classes=1,
        enable_multiscale_refine=False,
    ):
        super().__init__()
        self.enable_multiscale_refine = bool(enable_multiscale_refine)

        # 1. Projection layer (SiLU + GroupNorm)
        self.z_proj = nn.Sequential(
            nn.Conv2d(z_dim, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, base_channels),
            nn.SiLU(inplace=True),
        )

        # Project f3 to a lower-dimensional representation
        # because it may contain a large number of channels.
        self.f3_proj = nn.Sequential(
            nn.Conv2d(f3_dim, base_channels, 1, bias=False),
            nn.GroupNorm(32, base_channels),
            nn.SiLU(inplace=True),
        )

        # 2. Core feature processing after fusion: use ASPP instead of
        # a simple convolution.
        # The input channel count is base_channels for element-wise addition
        # or 2 * base_channels for concatenation.
        # Concatenation is recommended because it preserves more information,
        # although it requires slightly more GPU memory.
        self.aspp = ASPP(base_channels * 2, base_channels)
        self.multiscale_refine = (
            MultiScaleResidualRefinement(
                mid_channels=640,
                high_channels=320,
                out_channels=base_channels,
            )
            if self.enable_multiscale_refine
            else None
        )

        # 3. Decoder using ResidualUpBlock
        self.up1 = ResidualUpBlock(base_channels, base_channels // 2)
        self.up2 = ResidualUpBlock(base_channels // 2, base_channels // 4)
        self.up3 = ResidualUpBlock(base_channels // 4, 32)  # Output 32 channels

        # 4. Classifier
        self.classifier = nn.Conv2d(32, num_classes, 1)

        # Initialize the output bias to reduce the risk of exploding
        # loss during the early stages of training.
        nn.init.constant_(self.classifier.bias, -3.0)

    def forward(self, z, f3_dict, thermal_emb=None):
        # f3 can be either a dictionary or a tensor.
        f3 = f3_dict["mid"] if isinstance(f3_dict, dict) else f3_dict

        # Process z
        z_feat = self.z_proj(z)  # [B, 256, 60, 80]

        # Process f3 and resize it to match the spatial dimensions of z.
        # Some PyTorch/CUDA versions do not support bilinear upsampling
        # with BFloat16. Therefore, interpolation is performed in FP32,
        # after which the subsequent projection layer restores the
        # computation dtype under autocast.
        f3_feat = F.interpolate(
            f3.float(),
            size=z_feat.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        f3_feat = self.f3_proj(f3_feat)  # [B, 256, 60, 80]
        if f3_feat.dtype != z_feat.dtype:
            f3_feat = f3_feat.to(dtype=z_feat.dtype)

        # --- Key improvement: concatenation-based fusion ---
        # Unlike z + f3, concatenation allows the network to learn
        # how to balance the semantic information in z against
        # the fine-grained details in f3.
        x = torch.cat([z_feat, f3_feat], dim=1)  # [B, 512, 60, 80]

        # Core feature extraction
        x = self.aspp(x)  # [B, 256, 60, 80]

        if self.enable_multiscale_refine:
            if not isinstance(f3_dict, dict) or len(f3_dict.get("up", [])) < 2:
                raise RuntimeError(
                    "Multi-scale refinement requires the last two UNet up-block features."
                )
            residual = self.multiscale_refine(
                f3_dict["up"][-2],
                f3_dict["up"][-1],
                output_size=x.shape[2:],
                output_dtype=x.dtype,
            )
            x = x + residual

        # Upsampling
        x = self.up1(x)  # -> 120x160
        x = self.up2(x)  # -> 240x320
        x = self.up3(x)  # -> 480x640

        return self.classifier(x)
