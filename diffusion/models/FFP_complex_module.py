import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_safe_groups(channels, max_groups=8):
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1




# Try to import DeformConv2d from torchvision.ops; fallback to conv if unavailable
try:
    from torchvision.ops import DeformConv2d
    _HAS_DEFORM = True
except Exception:
    DeformConv2d = None
    _HAS_DEFORM = False
    warnings.warn("torchvision.ops.DeformConv2d not available — using simulated Conv2d placeholder. "
                  "Install/upgrade torchvision (>=0.9) to use real DeformConv2d for best results.")


class UncertaintyAlignmentFusion(nn.Module):
    """
    Spatial alignment + residual-based uncertainty estimator using Deformable Conv (DCN).
    - channels: number of channels in f_rgb/f_th
    - kernel_size: kernel size for DCN (default 3)
    - max_offset: maximum allowed offset magnitude in pixels (clamped)
    """
    def __init__(self, channels, kernel_size=3, max_offset=15.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.max_offset = float(max_offset)

        # offset predictor: output 2 * kH * kW channels (format required by DeformConv2d)
        out_offset_channels = 2 * kernel_size * kernel_size
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_offset_channels, kernel_size=3, padding=1)
        )
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

        # DeformConv2d or simulated fallback
        if _HAS_DEFORM:
            # DeformConv2d(in_channels, out_channels, kernel_size, padding=..., bias=True/False)
            # we'll use same in/out channels and padding appropriate for kernel_size
            padding = kernel_size // 2
            self.dcn = DeformConv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True)
        else:
            # fallback: regular conv that preserves channels and spatial dims
            self.dcn = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)

        # uncertainty estimator: take channel-aggregated residual (1 channel) -> 0..1
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(1, max(4, channels // 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // 2), 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, f_rgb, f_th):
        """
        Args:
            f_rgb, f_th: [B, C, H, W]
        Returns:
            f_th_aligned: [B, C, H, W]
            uncertainty_map: [B, 1, H, W] in [0,1] (higher -> more uncertain)
            optional: offset (for visualization) shape [B, 2*kH*kW, H, W]
        """
        combined = torch.cat([f_rgb, f_th], dim=1)  # [B, 2C, H, W]
        offset_raw = self.offset_conv(combined)     # [B, 2*k*k, H, W]

        # Scale & clamp offset:
        # offset_raw may be unbounded; map to [-max_offset, max_offset]
        # we use tanh to squish then multiply by max_offset; also clamp to be extra safe
        
        offset = torch.tanh(offset_raw/4) * 3
        #offset = torch.tanh(offset_raw) * self.max_offset
        offset = torch.clamp(offset, -self.max_offset, self.max_offset)

        # Apply DCN if available, else fallback conv (offset is unused for fallback)
        if _HAS_DEFORM:
            # DeformConv2d expects offset in shape [B, 2*kH*kW, H_out, W_out]
            f_th_aligned = self.dcn(f_th, offset)
        else:
            # fallback path: use regular conv (offset ignored)
            f_th_aligned = self.dcn(f_th)

        # residual-based uncertainty: channel-aggregated absolute diff
        diff = torch.abs(f_rgb - f_th_aligned)        # [B,C,H,W]
        diff_ch = diff.mean(dim=1, keepdim=True)      # [B,1,H,W]
        uncertainty_map = self.uncertainty_estimator(diff_ch)  # [B,1,H,W], values in [0,1]

        return f_th_aligned, uncertainty_map, offset


class StrictComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bias_re = nn.Parameter(torch.zeros(out_channels))
        self.bias_im = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x_real, x_imag):
        out_real = self.conv_re(x_real) - self.conv_im(x_imag)
        out_imag = self.conv_im(x_real) + self.conv_re(x_imag)
        out_real = out_real + self.bias_re.view(1, -1, 1, 1)
        out_imag = out_imag + self.bias_im.view(1, -1, 1, 1)
        return out_real, out_imag


class ComplexModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = 1e-6

    def forward(self, re, im):
        amp = torch.sqrt(re * re + im * im + self.eps)
        amp_out = F.relu(amp + self.b)
        scale = amp_out / (amp + self.eps)
        return re * scale, im * scale


class UltimateFrequencyFusionV3(nn.Module):
    """
    UltimateFrequencyFusionV3 (Thermal-Guided, with DCN-based spatial alignment)
    Key features:
      - spatial coarse alignment via DeformConv2d (offset clamped)
      - residual-based uncertainty from alignment -> used to re-estimate var_th in freq domain
      - UG-PRL: phase residual learning (aligner)
      - Thermal-guided gate with learnable gate_scale (trainable)
      - keeps rfft2/irfft2 cpu->device pattern as requested
    """
    def __init__(self, channels=64,
                 phase_loss_mode='monitor',
                 uncertainty_smooth=True,
                 var_smooth_ksize=3,
                 uncertainty_freq_scale=3.0,
                 dcn_kernel=3,
                 dcn_max_offset=6.0):
        super().__init__()
        self.C = channels
        self.phase_loss_mode = phase_loss_mode
        self.var_smooth_ksize = var_smooth_ksize
        self.uncertainty_freq_scale = float(uncertainty_freq_scale)

        # alignment module (DCN)
        self.align_module = UncertaintyAlignmentFusion(channels, kernel_size=dcn_kernel, max_offset=dcn_max_offset)

        # complex encoder & actuator
        self.complex_encoder = StrictComplexConv2d(channels * 2, channels, kernel_size=1)
        self.complex_act = ComplexModReLU(channels)
        self.complex_res_proj_re = nn.Conv2d(channels, channels, kernel_size=1)
        self.complex_res_proj_im = nn.Conv2d(channels, channels, kernel_size=1)

        # phys encoder
        phys_in = 2 * channels * 2 + 2
        self.phys_encoder = nn.Sequential(
            nn.Conv2d(phys_in, max(8, channels * 4), kernel_size=1),
            nn.GroupNorm(get_safe_groups(max(8, channels * 4)), max(8, channels * 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, channels * 4), max(8, channels * 2), kernel_size=1),
            nn.GroupNorm(get_safe_groups(max(8, channels * 2)), max(8, channels * 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, channels * 2), channels, kernel_size=1)
        )

        # uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, kernel_size=1),
            nn.GroupNorm(get_safe_groups(channels * 2), channels * 2)
        )
        self.uncertainty_smooth = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1,
                                            groups=channels * 2, bias=False) if uncertainty_smooth else None
        self.var_smooth_pool = nn.AvgPool2d(var_smooth_ksize, stride=1,
                                            padding=var_smooth_ksize // 2) if var_smooth_ksize and var_smooth_ksize > 1 else None

        # phase aligner & gating & refine
        self.phase_aligner = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(get_safe_groups(channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.Tanh()
        )
        self.align_scale = nn.Parameter(torch.tensor(0.5))

        # learnable gate scale
        self.gate_scale = nn.Parameter(torch.tensor(1.0))

        # residual gate
        self.res_gate = nn.Parameter(torch.tensor(0.0))

        # spatial refine
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(get_safe_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.gate_beta = nn.Parameter(torch.tensor(-2.0))

    def get_frequency_coords(self, H, W_freq, device):
        y = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(1, 1, H, W_freq)
        x = torch.linspace(-1, 1, W_freq, device=device).view(1, 1, 1, W_freq).expand(1, 1, H, W_freq)
        return torch.cat([x, y], dim=1)

    def forward(self, x_rgb, x_th):
        """
        Inputs:
            x_rgb, x_th: [B, C, H, W] spatial features (e.g., pr, pt from ResNet layer2)
        Returns:
            x_final: [B, C, H, W] fused spatial output
            align_loss: scalar tensor
            monitor: dict
        """
        device = x_rgb.device
        B, C, H, W = x_rgb.shape
        eps = 1e-12

        # 0) Spatial coarse alignment (DCN) + residual-based uncertainty
        f_th_aligned, uncertainty_map, offset = self.align_module(x_rgb, x_th)
        uncertainty_mean = uncertainty_map.mean().detach()
        


        loss_align_spatial = torch.nn.functional.l1_loss(f_th_aligned, x_rgb)

        # 1) FFT -> complex (keeps cpu() pattern as requested)
        z_rgb = torch.fft.rfft2(x_rgb.float().cpu(), norm='ortho').to(device)
        z_th = torch.fft.rfft2(f_th_aligned.float().cpu(), norm='ortho').to(device)

        # 2) complex encoder (real/imag)
        in_real = torch.cat([z_rgb.real, z_th.real], dim=1)
        in_imag = torch.cat([z_rgb.imag, z_th.imag], dim=1)
        feat_re, feat_im = self.complex_encoder(in_real, in_imag)
        feat_re, feat_im = self.complex_act(feat_re, feat_im)
        corr_re = self.complex_res_proj_re(feat_re)
        corr_im = self.complex_res_proj_im(feat_im)

        # 3) phys features
        amp_rgb = z_rgb.abs()
        amp_th = z_th.abs()
        phase_delta = z_rgb.angle() - z_th.angle()
        phase_sin = torch.sin(phase_delta)
        phase_cos = torch.cos(phase_delta)
        W_freq = z_rgb.shape[-1]
        coords = self.get_frequency_coords(H, W_freq, device).expand(B, -1, -1, -1)
        phys_input = torch.cat([amp_rgb, amp_th, phase_sin, phase_cos, coords], dim=1)
        phys_feat = self.phys_encoder(phys_input)

        # 4) uncertainty head -> initial var_rgb, var_th
        joint_feat = torch.cat([feat_re, feat_im, phys_feat], dim=1)
        raw_log_vars = self.uncertainty_head(joint_feat)
        if self.uncertainty_smooth is not None:
            raw_log_vars = self.uncertainty_smooth(raw_log_vars)
        log_var_rgb, log_var_th = torch.chunk(raw_log_vars, 2, dim=1)
        var_rgb = F.softplus(log_var_rgb).clamp(min=1e-6, max=1e6)
        var_th = F.softplus(log_var_th).clamp(min=1e-6, max=1e6)
        if self.var_smooth_pool is not None:
            var_rgb = self.var_smooth_pool(var_rgb)
            var_th = self.var_smooth_pool(var_th)

        # 5) Re-estimate var_th using residual-based uncertainty (freq domain)
        #    transform uncertainty_map -> freq magnitude and inflate var_th where alignment failed.
        u_fft = torch.fft.rfft2(uncertainty_map.float().cpu(), norm='ortho').abs().to(device)  # [B,1,H,W_freq]
        u_fft_exp = u_fft.expand(-1, var_th.shape[1], -1, -1)
        var_th = var_th * (1.0 + self.uncertainty_freq_scale * u_fft_exp)

        # 6) amplitude fusion (uncertainty-weighted)
        w_rgb = 1.0 / (var_rgb + eps)
        w_th = 1.0 / (var_th + eps)
        w_sum = w_rgb + w_th + eps
        amp_fused = (w_rgb * amp_rgb + w_th * amp_th) / w_sum

        rgb_ratio = w_rgb / (w_rgb + w_th + eps)


        # 7) phase residual learning (UG-PRL)
        align_feat = self.phase_aligner(phys_feat)
        corr_from_aligner_re = self.align_scale * align_feat
        corr_from_aligner_im = torch.zeros_like(corr_from_aligner_re, device=device)
        corr_re_total = corr_re + corr_from_aligner_re
        corr_im_total = corr_im + corr_from_aligner_im
        z_residual = torch.complex(corr_re_total, corr_im_total)
        z_th_shifted = z_th + z_residual

        # 8) unit phase vectors & Thermal-guided Gate w/ learnable scale
        u_rgb = z_rgb / (z_rgb.abs() + eps)
        u_th_aligned = z_th_shifted / (z_th_shifted.abs() + eps)

        diff = torch.log(var_th + eps) - torch.log(var_rgb + eps)
        bias = 0.0               # thermal-guided default (unbiased)
        temperature = 1.0
        gate_input = (diff + bias) * temperature * self.gate_scale
        gate = torch.sigmoid(gate_input)

        # clamp to keep some RGB gradient path while allowing Thermal dominance
        gate = torch.clamp(gate, 0.1, 0.9)

        # phase interpolation in complex plane
        u_fused = gate * u_rgb + (1.0 - gate) * u_th_aligned
        u_fused = u_fused / (u_fused.abs() + eps)

        # 9) rebuild fused spectrum + residual gating
        z_final = torch.complex(amp_fused * u_fused.real, amp_fused * u_fused.imag)
        res_gate_val = torch.sigmoid(self.res_gate)
        z_final = z_final + res_gate_val * z_residual

        # 10) IRFFT -> spatial domain (retain cpu->device)
        x_fused = torch.fft.irfft2(z_final.cpu(), s=(H, W), norm='ortho').to(device)

        # spatial refine
        residual = self.spatial_refine(x_fused)
        alpha = torch.sigmoid(self.gate_beta)
        x_final = x_fused + alpha * residual

        # 11) align_loss & monitor
        cos_phase = (u_rgb.real * u_th_aligned.real + u_rgb.imag * u_th_aligned.imag)
        phase_dist = 1.0 - cos_phase
        align_weight = 0.2 + 0.8 * gate.detach()
        align_loss = (align_weight * phase_dist).mean()



        monitor = {
            'w_rgb_mean': float(w_rgb.mean().detach().cpu()),
            'w_th_mean': float(w_th.mean().detach().cpu()),
            'gate_mean': float(gate.mean().detach().cpu()),
            'res_gate': float(res_gate_val.detach().cpu()),
            'align_scale': float(self.align_scale.detach().cpu()),
            'gate_scale': float(self.gate_scale.detach().cpu()),
            'uncertainty_mean': float(uncertainty_mean.cpu()),
            'rgb_ratio_mean': float(rgb_ratio.mean().detach().cpu()),
            'var_rgb_mean': float(var_rgb.mean().detach().cpu()),
            'var_th_mean': float(var_th.mean().detach().cpu()),


        }

        debug = {
            "amp_rgb": amp_rgb,
            "amp_th": amp_th,
            "amp_fused": amp_fused,
            "offset": offset,
            "feat_rgb": x_rgb,                  # feature before fusion
            "feat_th": x_th,                    # thermal feature before align
            "feat_th_aligned": f_th_aligned,    # aligned thermal feature
        }
        
        

        # also return offset for debugging / visualization if needed
        return x_final, align_loss, monitor, debug,loss_align_spatial 

