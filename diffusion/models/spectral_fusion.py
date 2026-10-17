"""SpecDiff R-ASE: coarse alignment, residual-frequency risk, and spectral fusion.

The parameter layout is retained for strict loading of existing R-ASE weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


def get_safe_groups(channels, max_groups=8):
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


class UncertaintyAlignmentFusion(nn.Module):
    """Coarse Thermal-to-RGB alignment plus residual discrepancy estimation.

    A bounded raw feature discrepancy supplies the main residual cue,
    with a learned estimator providing an additional calibration term.
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        max_offset=15.0,
        learned_uncertainty_mix=0.10,
        offset_temperature=4.0,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.max_offset = float(max_offset)
        self.learned_uncertainty_mix = float(learned_uncertainty_mix)
        self.offset_temperature = float(offset_temperature)

        if not 0.0 <= self.learned_uncertainty_mix <= 1.0:
            raise ValueError("learned_uncertainty_mix must be in [0, 1]")
        if self.offset_temperature <= 0:
            raise ValueError("offset_temperature must be positive")

        # Offset predictor: 2 * kH * kW channels as required by DeformConv2d.
        out_offset_channels = 2 * kernel_size * kernel_size
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_offset_channels, kernel_size=3, padding=1),
        )
        # Start from zero displacement.
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

        padding = kernel_size // 2
        self.dcn = DeformConv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, bias=True
        )

        # Conservative identity initialization. Existing checkpoints overwrite it.
        with torch.no_grad():
            try:
                nn.init.dirac_(self.dcn.weight)
            except Exception:
                pass
            if getattr(self.dcn, "bias", None) is not None:
                nn.init.zeros_(self.dcn.bias)

        # Keep the original learned estimator and its parameter layout for strict
        # checkpoint compatibility. It now acts as a small calibration branch.
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(1, max(4, channels // 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // 2), 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _monotonic_bound(residual):
        """Map non-negative residual to [0,1) without per-sample normalization.

        r/(1+r) is strictly monotonic, preserves cross-severity ordering, and
        avoids the severity information loss caused by min-max normalization.
        """
        residual = residual.clamp_min(0.0)
        return residual / (1.0 + residual)

    def forward(self, f_rgb, f_th):
        """
        Args:
            f_rgb, f_th: [B, C, H, W]

        Returns:
            f_th_aligned: [B, C, H, W]
            discrepancy_map: [B, 1, H, W], larger -> less reliable
            offset: [B, 2*kH*kW, H, W]
        """
        combined = torch.cat([f_rgb, f_th], dim=1)
        offset_raw = self.offset_conv(combined)

        # Use the configured maximum instead of the old hard-coded *3. The
        # temperature keeps early gradients moderate while allowing the full
        # range when needed.
        offset = torch.tanh(offset_raw / self.offset_temperature) * self.max_offset

        f_th_aligned = self.dcn(f_th, offset)

        discrepancy_map = self.discrepancy_from_features(f_rgb, f_th_aligned)

        return f_th_aligned, discrepancy_map, offset

    def discrepancy_from_features(self, f_rgb, f_th):
        """Estimate uncertainty from a supplied thermal feature pair.

        This keeps uncertainty estimation identical when deformable sampling
        is disabled, so the ablation isolates the alignment stage.
        """
        raw_residual = (f_rgb - f_th).abs().mean(dim=1, keepdim=True)
        monotonic_map = self._monotonic_bound(raw_residual)

        # A log-compressed input makes the calibration branch less sensitive to
        # isolated feature outliers. Its contribution is intentionally small.
        learned_map = self.uncertainty_estimator(torch.log1p(raw_residual))
        mix = self.learned_uncertainty_mix
        discrepancy_map = (1.0 - mix) * monotonic_map + mix * learned_map
        discrepancy_map = discrepancy_map.clamp(0.0, 1.0)

        return discrepancy_map


class StrictComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv_re = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.conv_im = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
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
    """Reliability-aware RGB-T spectral fusion.

    The trainable parameter layout is intentionally kept compatible with the
    original version. New controls are scalar Python attributes only.
    """

    def __init__(
        self,
        channels=64,
        phase_loss_mode="monitor",
        uncertainty_smooth=True,
        var_smooth_ksize=3,
        uncertainty_freq_scale=2.0,
        dcn_kernel=3,
        dcn_max_offset=6.0,
        learned_uncertainty_mix=0.10,
        uncertainty_norm_residual_scale=0.25,
        uncertainty_smooth_residual_scale=0.25,
        frequency_local_mix=0.50,
        frequency_shape_clip=5.0,
        phase_gate_min=0.05,
        phase_gate_max=0.95,
        phase_gate_form="bounded_log_ratio",
        phase_residual_ratio=0.50,
        residual_injection_max=0.25,
        disable_deformable_alignment=False,
    ):
        super().__init__()
        self.C = channels
        self.phase_loss_mode = phase_loss_mode
        self.var_smooth_ksize = var_smooth_ksize
        self.uncertainty_freq_scale = float(uncertainty_freq_scale)
        self.uncertainty_norm_residual_scale = float(uncertainty_norm_residual_scale)
        self.uncertainty_smooth_residual_scale = float(
            uncertainty_smooth_residual_scale
        )
        self.frequency_local_mix = float(frequency_local_mix)
        self.frequency_shape_clip = float(frequency_shape_clip)
        self.phase_gate_min = float(phase_gate_min)
        self.phase_gate_max = float(phase_gate_max)
        self.phase_gate_form = str(phase_gate_form)
        self.phase_residual_ratio = float(phase_residual_ratio)
        self.residual_injection_max = float(residual_injection_max)
        self.disable_deformable_alignment = bool(disable_deformable_alignment)

        if self.uncertainty_freq_scale < 0:
            raise ValueError("uncertainty_freq_scale must be nonnegative")
        if not 0.0 <= self.frequency_local_mix <= 1.0:
            raise ValueError("frequency_local_mix must be in [0,1]")
        if not 0.0 <= self.phase_gate_min < self.phase_gate_max <= 1.0:
            raise ValueError("Require 0 <= phase_gate_min < phase_gate_max <= 1")
        if self.phase_gate_form not in {
            "equal",
            "bounded_difference",
            "bounded_log_ratio",
        }:
            raise ValueError(f"Unknown phase_gate_form: {self.phase_gate_form}")
        if self.phase_residual_ratio <= 0:
            raise ValueError("phase_residual_ratio must be positive")
        if not 0.0 < self.residual_injection_max <= 1.0:
            raise ValueError("residual_injection_max must be in (0,1]")

        # 0) Coarse spatial alignment + local discrepancy cue.
        self.align_module = UncertaintyAlignmentFusion(
            channels,
            kernel_size=dcn_kernel,
            max_offset=dcn_max_offset,
            learned_uncertainty_mix=learned_uncertainty_mix,
        )

        # 1) Learned complex spectral interactions.
        self.complex_encoder = StrictComplexConv2d(
            channels * 2, channels, kernel_size=1
        )
        self.complex_act = ComplexModReLU(channels)
        self.complex_res_proj_re = nn.Conv2d(channels, channels, kernel_size=1)
        self.complex_res_proj_im = nn.Conv2d(channels, channels, kernel_size=1)

        # Conservative fresh-start initialization: no large spectral correction.
        nn.init.zeros_(self.complex_res_proj_re.weight)
        nn.init.zeros_(self.complex_res_proj_re.bias)
        nn.init.zeros_(self.complex_res_proj_im.weight)
        nn.init.zeros_(self.complex_res_proj_im.bias)

        # 2) Interpretable spectral cues: log-amplitude, phase sin/cos, coordinates.
        phys_in = 2 * channels * 2 + 2
        self.phys_encoder = nn.Sequential(
            nn.Conv2d(phys_in, max(8, channels * 4), kernel_size=1),
            nn.GroupNorm(get_safe_groups(max(8, channels * 4)), max(8, channels * 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, channels * 4), max(8, channels * 2), kernel_size=1),
            nn.GroupNorm(get_safe_groups(max(8, channels * 2)), max(8, channels * 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, channels * 2), channels, kernel_size=1),
        )

        # 3) Frequency-dependent modality uncertainty.
        # Keep the exact original submodules/keys for strict checkpoint loading.
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, kernel_size=1),
            nn.GroupNorm(get_safe_groups(channels * 2), channels * 2),
        )
        self.uncertainty_smooth = (
            nn.Conv2d(
                channels * 2,
                channels * 2,
                kernel_size=3,
                padding=1,
                groups=channels * 2,
                bias=False,
            )
            if uncertainty_smooth
            else None
        )
        if self.uncertainty_smooth is not None:
            # Treat smoothing as a learnable residual correction; start neutral.
            nn.init.zeros_(self.uncertainty_smooth.weight)

        self.var_smooth_pool = (
            nn.AvgPool2d(
                var_smooth_ksize,
                stride=1,
                padding=var_smooth_ksize // 2,
            )
            if var_smooth_ksize and var_smooth_ksize > 1
            else None
        )

        # 4) Phase refinement, gating, and spatial refinement.
        self.phase_aligner = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(get_safe_groups(channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )
        # Fresh model starts without additional phase correction.
        nn.init.zeros_(self.phase_aligner[3].weight)
        nn.init.zeros_(self.phase_aligner[3].bias)

        self.align_scale = nn.Parameter(torch.tensor(0.5))
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.res_gate = nn.Parameter(torch.tensor(0.0))

        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(get_safe_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.spatial_refine[-1].weight)
        nn.init.zeros_(self.spatial_refine[-1].bias)
        self.gate_beta = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _bounded_variance(log_var):
        # 1e-4 avoids extreme inverse weights while retaining a wide dynamic range.
        return F.softplus(log_var).clamp(min=1e-4, max=1e4)

    @staticmethod
    def _mean_channel_residual(x_rgb, x_th_aligned):
        return (x_rgb - x_th_aligned).abs().mean(dim=1, keepdim=True)

    @staticmethod
    def _bound_phase_residual(z_residual, reference_amplitude, ratio, eps=1e-8):
        """Bound correction magnitude relative to local Thermal spectrum amplitude."""
        res_amp = z_residual.abs()
        max_amp = ratio * reference_amplitude.detach().clamp_min(eps)
        scale = torch.clamp(max_amp / (res_amp + eps), max=1.0)
        return z_residual * scale

    def get_frequency_coords(self, H, W_freq, device, dtype=torch.float32):
        y = (
            torch.linspace(-1, 1, H, device=device, dtype=dtype)
            .view(1, 1, H, 1)
            .expand(1, 1, H, W_freq)
        )
        x = (
            torch.linspace(-1, 1, W_freq, device=device, dtype=dtype)
            .view(1, 1, 1, W_freq)
            .expand(1, 1, H, W_freq)
        )
        return torch.cat([x, y], dim=1)

    def _residual_frequency_risk(self, raw_residual, eps=1e-6):
        """Build severity-preserving frequency risk from raw spatial residual.

        global_risk keeps degradation magnitude. The centered FFT describes where
        the residual energy lies in frequency. We normalize only the spectral
        shape and then re-scale it by global_risk, so increasing degradation is
        not erased by per-sample normalization.
        """
        bounded = raw_residual.clamp_min(0.0) / (1.0 + raw_residual.clamp_min(0.0))
        global_risk = bounded.mean(dim=(2, 3), keepdim=True)  # [B,1,1,1]

        centered = raw_residual - raw_residual.mean(dim=(2, 3), keepdim=True)
        local_spectrum = torch.fft.rfft2(centered.float(), norm="ortho").abs()
        local_spectrum = torch.log1p(local_spectrum)
        local_mean = local_spectrum.mean(dim=(2, 3), keepdim=True).clamp_min(eps)
        local_shape = (local_spectrum / local_mean).clamp(
            0.0, self.frequency_shape_clip
        )

        mix = self.frequency_local_mix
        # Mean risk remains approximately global_risk; local_shape redistributes it.
        freq_risk = global_risk * ((1.0 - mix) + mix * local_shape)
        return freq_risk, global_risk, local_spectrum

    def forward(self, x_rgb, x_th, rgb_quality_delta=None, return_diagnostics=False):
        """
        Inputs:
            x_rgb, x_th: [B, C, H, W] spatial features.

        Returns:
            x_final: [B, C, H, W]
            align_loss: scalar
            monitor: dict of lightweight scalars
            debug: dict of tensors
            loss_align_spatial: scalar
        """
        device = x_rgb.device
        B, C, H, W = x_rgb.shape
        eps = 1e-8

        # ------------------------------------------------------------------
        # 0) Coarse alignment + spatial discrepancy
        # ------------------------------------------------------------------
        if self.disable_deformable_alignment:
            # Feed raw, unwarped thermal features to the unchanged uncertainty
            # and spectral branches. Only deformable sampling is removed.
            f_th_aligned = x_th
            spatial_discrepancy = self.align_module.discrepancy_from_features(
                x_rgb, f_th_aligned
            )
            offset = torch.zeros(
                B,
                2 * self.align_module.kernel_size * self.align_module.kernel_size,
                H,
                W,
                device=x_th.device,
                dtype=x_th.dtype,
            )
        else:
            f_th_aligned, spatial_discrepancy, offset = self.align_module(x_rgb, x_th)
        raw_feature_residual = self._mean_channel_residual(x_rgb, f_th_aligned)
        discrepancy_mean = spatial_discrepancy.mean().detach()

        # Cross-modal absolute feature values need not be identical. A cosine
        # structural objective is less likely to collapse modality-specific
        # intensity/response statistics than raw L1.
        rgb_unit_spatial = F.normalize(x_rgb.float(), dim=1, eps=1e-6)
        th_unit_spatial = F.normalize(f_th_aligned.float(), dim=1, eps=1e-6)
        spatial_cos = (rgb_unit_spatial * th_unit_spatial).sum(dim=1)
        # Do not retain a spatial-alignment surrogate in the no-alignment
        # ablation; otherwise the removed stage would still be supervised.
        loss_align_spatial = (
            torch.zeros((), device=x_rgb.device, dtype=x_rgb.dtype)
            if self.disable_deformable_alignment
            else (1.0 - spatial_cos).mean()
        )

        # ------------------------------------------------------------------
        # 1) FFT
        # ------------------------------------------------------------------
        z_rgb = torch.fft.rfft2(x_rgb.float(), norm="ortho")
        z_th = torch.fft.rfft2(f_th_aligned.float(), norm="ortho")

        # ------------------------------------------------------------------
        # 2) Learned complex spectral interactions
        # ------------------------------------------------------------------
        in_real = torch.cat([z_rgb.real, z_th.real], dim=1)
        in_imag = torch.cat([z_rgb.imag, z_th.imag], dim=1)
        feat_re, feat_im = self.complex_encoder(in_real, in_imag)
        feat_re, feat_im = self.complex_act(feat_re, feat_im)
        corr_re = self.complex_res_proj_re(feat_re)
        corr_im = self.complex_res_proj_im(feat_im)

        # ------------------------------------------------------------------
        # 3) Interpretable spectral cues
        # ------------------------------------------------------------------
        amp_rgb = z_rgb.abs()
        amp_th = z_th.abs()
        phase_delta = z_rgb.angle() - z_th.angle()
        phase_sin = torch.sin(phase_delta)
        phase_cos = torch.cos(phase_delta)

        W_freq = z_rgb.shape[-1]
        coords = self.get_frequency_coords(H, W_freq, device, dtype=amp_rgb.dtype)
        coords = coords.expand(B, -1, -1, -1)

        # Log compression prevents a few very large low-frequency amplitudes from
        # dominating uncertainty estimation while preserving frequency ordering.
        phys_input = torch.cat(
            [
                torch.log1p(amp_rgb),
                torch.log1p(amp_th),
                phase_sin,
                phase_cos,
                coords,
            ],
            dim=1,
        )
        phys_feat = self.phys_encoder(phys_input)

        # ------------------------------------------------------------------
        # 4) Initial frequency-dependent uncertainty
        # ------------------------------------------------------------------
        joint_feat = torch.cat([feat_re, feat_im, phys_feat], dim=1)

        # IMPORTANT: the old implementation ended the uncertainty head with
        # GroupNorm. Because GN normalizes each sample, global degradation-scale
        # changes can be attenuated. Preserve the unnormalized Conv output as the
        # severity-carrying base and use the existing GN only as a residual shape
        # correction. No parameter keys are changed.
        uncertainty_base = self.uncertainty_head[0](joint_feat)
        uncertainty_shape = self.uncertainty_head[1](uncertainty_base)
        raw_log_vars = (
            uncertainty_base + self.uncertainty_norm_residual_scale * uncertainty_shape
        )

        if self.uncertainty_smooth is not None:
            raw_log_vars = (
                raw_log_vars
                + self.uncertainty_smooth_residual_scale
                * self.uncertainty_smooth(raw_log_vars)
            )

        log_var_rgb, log_var_th = torch.chunk(raw_log_vars, 2, dim=1)
        var_rgb = self._bounded_variance(log_var_rgb)
        var_th = self._bounded_variance(log_var_th)

        if self.var_smooth_pool is not None:
            var_rgb = self.var_smooth_pool(var_rgb)
            var_th = self.var_smooth_pool(var_th)

        # Keep the original spectral estimate for diagnostics, then apply an
        # optional image-level RGB quality correction.  This correction is
        # inserted before both amplitude weighting and phase gating, so the two
        # fusion decisions use the same corrected reliability estimate.
        var_rgb_base = var_rgb
        if rgb_quality_delta is not None:
            delta = rgb_quality_delta.to(device=var_rgb.device, dtype=var_rgb.dtype)
            var_rgb = (var_rgb * torch.exp(delta)).clamp(min=1e-6, max=1e6)
        var_rgb_pre = var_rgb
        var_th_pre = var_th

        # ------------------------------------------------------------------
        # 5) Residual-aware Thermal uncertainty modulation
        # ------------------------------------------------------------------
        freq_risk, global_risk, discrepancy_spectrum = self._residual_frequency_risk(
            raw_feature_residual
        )
        freq_risk_exp = freq_risk.expand(-1, var_th.shape[1], -1, -1)

        # Strictly positive multiplicative inflation. High residual severity and
        # high-risk frequencies both increase Thermal uncertainty.
        var_th = var_th_pre * (1.0 + self.uncertainty_freq_scale * freq_risk_exp)
        var_th = var_th.clamp(min=1e-4, max=1e4)

        # ------------------------------------------------------------------
        # 6) Inverse-uncertainty amplitude fusion
        # ------------------------------------------------------------------
        w_rgb = 1.0 / (var_rgb + eps)
        w_th = 1.0 / (var_th + eps)
        w_sum = w_rgb + w_th + eps
        amp_fused = (w_rgb * amp_rgb + w_th * amp_th) / w_sum
        rgb_ratio = w_rgb / w_sum
        thermal_ratio = w_th / w_sum

        # ------------------------------------------------------------------
        # 7) Conservative learned phase residual
        # ------------------------------------------------------------------
        align_feat = self.phase_aligner(phys_feat)
        # Bound the scalar while preserving the original parameter.
        align_scale_eff = torch.tanh(self.align_scale)
        corr_from_aligner_re = align_scale_eff * align_feat
        corr_from_aligner_im = torch.zeros_like(corr_from_aligner_re, device=device)

        corr_re_total = corr_re + corr_from_aligner_re
        corr_im_total = corr_im + corr_from_aligner_im
        z_residual = torch.complex(corr_re_total, corr_im_total)
        z_residual = self._bound_phase_residual(
            z_residual,
            amp_th,
            ratio=self.phase_residual_ratio,
            eps=eps,
        )
        z_th_shifted = z_th + z_residual

        # ------------------------------------------------------------------
        # 8) Uncertainty-monotonic phase gate
        # ------------------------------------------------------------------
        u_rgb = z_rgb / (amp_rgb + eps)
        u_th_aligned = z_th_shifted / (z_th_shifted.abs() + eps)

        gate_scale_eff = F.softplus(self.gate_scale) + 1e-4
        if self.phase_gate_form == "equal":
            gate = torch.full_like(var_rgb, 0.5)
        else:
            if self.phase_gate_form == "bounded_difference":
                # Dimensionless, bounded difference keeps the comparison stable
                # when uncertainty magnitudes vary across images.
                gate_input = (var_th - var_rgb) / (var_th + var_rgb + eps)
            else:
                gate_input = torch.log(var_th + eps) - torch.log(var_rgb + eps)
            # A positive scale preserves the intended reliability ordering.
            gate_raw = torch.sigmoid(gate_input * gate_scale_eff)
            gate = (
                self.phase_gate_min
                + (self.phase_gate_max - self.phase_gate_min) * gate_raw
            )

        u_fused = gate * u_rgb + (1.0 - gate) * u_th_aligned
        u_fused = u_fused / (u_fused.abs() + eps)

        # ------------------------------------------------------------------
        # 9) Rebuild fused spectrum + bounded residual injection
        # ------------------------------------------------------------------
        z_final = torch.complex(
            amp_fused * u_fused.real,
            amp_fused * u_fused.imag,
        )

        # Original sigmoid(0)=0.5 injected a large residual at initialization.
        # Keep the same trainable parameter but cap the actual injection.
        res_gate_val = self.residual_injection_max * torch.sigmoid(self.res_gate)
        z_final = z_final + res_gate_val * z_residual

        # ------------------------------------------------------------------
        # 10) IFFT + conservative spatial refinement
        # ------------------------------------------------------------------
        x_fused = torch.fft.irfft2(z_final, s=(H, W), norm="ortho")
        residual = self.spatial_refine(x_fused)
        alpha = torch.sigmoid(self.gate_beta)
        x_final = x_fused + alpha * residual

        # ------------------------------------------------------------------
        # 11) Phase alignment objective and diagnostics
        # ------------------------------------------------------------------
        cos_phase = u_rgb.real * u_th_aligned.real + u_rgb.imag * u_th_aligned.imag
        phase_dist = 1.0 - cos_phase

        # If gate -> RGB, Thermal phase contributes little, so forcing strong
        # RGB/Thermal phase agreement is unnecessary. Align more where Thermal is
        # actually trusted and participates in fusion.
        thermal_phase_confidence = (1.0 - gate).detach()
        align_weight = 0.2 + 0.8 * thermal_phase_confidence
        align_loss = (align_weight * phase_dist).mean()

        correction_ratio = (z_residual.abs() / (amp_th + eps)).clamp(
            max=self.phase_residual_ratio
        )

        monitor = {
            "w_rgb_mean": float(w_rgb.mean().detach().cpu()),
            "w_th_mean": float(w_th.mean().detach().cpu()),
            "gate_mean": float(gate.mean().detach().cpu()),
            "res_gate": float(res_gate_val.detach().cpu()),
            "align_scale": float(self.align_scale.detach().cpu()),
            "align_scale_effective": float(align_scale_eff.detach().cpu()),
            "gate_scale": float(self.gate_scale.detach().cpu()),
            "gate_scale_effective": float(gate_scale_eff.detach().cpu()),
            "uncertainty_mean": float(discrepancy_mean.cpu()),
            "u_rgb_base_mean": float(var_rgb_base.mean().detach().cpu()),
            "u_rgb_pre_mean": float(var_rgb_pre.mean().detach().cpu()),
            "rgb_quality_delta_mean": (
                float(rgb_quality_delta.mean().detach().cpu())
                if rgb_quality_delta is not None
                else 0.0
            ),
            "u_th_pre_mean": float(var_th_pre.mean().detach().cpu()),
            "u_th_post_mean": float(var_th.mean().detach().cpu()),
            "spatial_discrepancy_mean": float(discrepancy_mean.cpu()),
            "raw_residual_mean": float(raw_feature_residual.mean().detach().cpu()),
            "global_risk_mean": float(global_risk.mean().detach().cpu()),
            "rgb_ratio_mean": float(rgb_ratio.mean().detach().cpu()),
            "thermal_ratio_mean": float(thermal_ratio.mean().detach().cpu()),
            "var_rgb_mean": float(var_rgb.mean().detach().cpu()),
            "var_th_mean": float(var_th.mean().detach().cpu()),
            "phase_correction_ratio_mean": float(
                correction_ratio.mean().detach().cpu()
            ),
        }

        debug = {
            "amp_rgb": amp_rgb,
            "amp_th": amp_th,
            "amp_fused": amp_fused,
            "offset": offset,
            "feat_rgb": x_rgb,
            "feat_th": x_th,
            "feat_th_aligned": f_th_aligned,
        }

        if return_diagnostics:
            debug.update(
                {
                    # Spatial-domain quantities.
                    "spatial_discrepancy": spatial_discrepancy,
                    "raw_feature_residual": raw_feature_residual,
                    "global_discrepancy": global_risk,
                    # Frequency-indexed reliability quantities.
                    "u_rgb_base": var_rgb_base,
                    "u_rgb_pre": var_rgb_pre,
                    "rgb_quality_delta": (
                        rgb_quality_delta
                        if rgb_quality_delta is not None
                        else torch.zeros(
                            var_rgb.shape[0],
                            1,
                            1,
                            1,
                            device=var_rgb.device,
                            dtype=var_rgb.dtype,
                        )
                    ),
                    "u_th_pre": var_th_pre,
                    "u_th_post": var_th,
                    "rgb_amplitude_weight": rgb_ratio,
                    "thermal_amplitude_weight": thermal_ratio,
                    "rgb_phase_gate": gate,
                    "discrepancy_spectrum": discrepancy_spectrum,
                    "frequency_risk": freq_risk,
                    "phase_discrepancy": phase_dist,
                    "phase_correction_ratio": correction_ratio,
                }
            )

        return x_final, align_loss, monitor, debug, loss_align_spatial
