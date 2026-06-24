import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


############################################
# Global paper style & palette
############################################
PAPER_COLORS = {
    "rgb": "#3B5B92",       # deep blue
    "thermal": "#C44E52",   # deep red
    "fused": "#2A9D8F",     # teal green
    "neutral": "#B0B7C3",   # light gray-blue
    "edge": "#555555",      # light dark edge
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})


############################################
# Utilities
############################################
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_figure(fig, path_no_ext):
    fig.savefig(
        path_no_ext + ".png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    fig.savefig(
        path_no_ext + ".pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )


def safe_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def tensor_to_img(tensor, size=None):
    """
    tensor: [B,C,H,W] or [C,H,W]
    return normalized image in [0,1]
    """
    if tensor.ndim == 4:
        img = tensor[0].detach().cpu().numpy()
    else:
        img = tensor.detach().cpu().numpy()

    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    return img


def feature_to_map(x, reduce="mean"):
    """
    x: [B,C,H,W] -> [H,W]
    """
    feat = x[0].detach().cpu().float()
    if reduce == "mean":
        return feat.mean(0).numpy()
    elif reduce == "max":
        return feat.max(0).values.numpy()
    else:
        raise ValueError(f"Unsupported reduce: {reduce}")


def normalize_map(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def style_image_axis(ax, title=None, panel_tag=None, title_pad=4):
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title, pad=title_pad, fontweight="normal", fontsize=11)

    if panel_tag is not None:
        ax.text(
            0.03, 0.97, panel_tag,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11.5,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.08", fc=(1, 1, 1, 0.58), ec="none")
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color(PAPER_COLORS["edge"])


############################################
# Spectrum utilities
############################################
def get_shifted_amplitude_spectrum(amp, use_log=True):
    """
    amp: [B,C,H,W], from rFFT amplitude-like tensor
    returns channel-averaged shifted spectrum [H,W]
    """
    spec = torch.from_numpy(feature_to_map(amp, reduce="mean"))
    spec = torch.clamp(spec, min=0.0)
    spec = torch.fft.fftshift(spec)
    if use_log:
        spec = torch.log1p(spec)
    return spec.numpy()


def get_shifted_power_spectrum(amp, use_log=False):
    spec = torch.from_numpy(feature_to_map(amp, reduce="mean"))
    spec = torch.clamp(spec, min=0.0)
    spec = spec ** 2
    spec = torch.fft.fftshift(spec)
    if use_log:
        spec = torch.log1p(spec)
    return spec.numpy()


def radial_profile(spec):
    H, W = spec.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = r.astype(np.int32)

    tbin = np.bincount(r_int.ravel(), weights=spec.ravel())
    nr = np.bincount(r_int.ravel())
    radial_mean = tbin / (nr + 1e-8)

    max_r = len(radial_mean) - 1
    r_norm = np.arange(len(radial_mean), dtype=np.float32) / max(max_r, 1)
    return r_norm, radial_mean


def spectral_distance(spec1, spec2, mode="l1"):
    if mode == "l1":
        return np.mean(np.abs(spec1 - spec2))
    elif mode == "l2":
        return np.sqrt(np.mean((spec1 - spec2) ** 2))
    else:
        raise ValueError(f"Unsupported mode: {mode}")


############################################
# Figure 1: Shifted log-amplitude spectra
############################################
def visualize_spectrum(rgb_amp, th_amp, fused_amp, save_dir):
    rgb = get_shifted_amplitude_spectrum(rgb_amp, use_log=True)
    th = get_shifted_amplitude_spectrum(th_amp, use_log=True)
    fused = get_shifted_amplitude_spectrum(fused_amp, use_log=True)

    vmin = min(rgb.min(), th.min(), fused.min())
    vmax = max(rgb.max(), th.max(), fused.max())

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.035, right=0.895, top=0.90, bottom=0.08, wspace=0.07)

    titles = ["RGB", "Thermal", "Fused"]
    panel_tags = ["(a)", "(b)", "(c)"]
    ims = []

    for i, (ax, spec, title) in enumerate(zip(axes, [rgb, th, fused], titles)):
        im = ax.imshow(
            spec, cmap="magma", vmin=vmin, vmax=vmax,
            interpolation="nearest", rasterized=True
        )
        ims.append(im)
        style_image_axis(ax, title=title, panel_tag=panel_tags[i])

    cax = fig.add_axes([0.905, 0.16, 0.013, 0.68])
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.set_label("Log Amplitude", labelpad=6)
    cbar.ax.tick_params(labelsize=9, width=0.8, length=3)
    cbar.outline.set_linewidth(0.75)

    save_figure(fig, os.path.join(save_dir, "fig1_spectrum"))
    plt.close(fig)


############################################
# Figure 2: Spectrum difference maps
############################################
def visualize_spectrum_difference(rgb_amp, th_amp, fused_amp, save_dir, fixed_vmax=None):
    rgb = get_shifted_amplitude_spectrum(rgb_amp, use_log=True)
    th = get_shifted_amplitude_spectrum(th_amp, use_log=True)
    fused = get_shifted_amplitude_spectrum(fused_amp, use_log=True)

    diff_rth = np.abs(rgb - th)
    diff_frgb = np.abs(fused - rgb)
    diff_fth = np.abs(fused - th)

    vmax = fixed_vmax if fixed_vmax is not None else max(
        diff_rth.max(), diff_frgb.max(), diff_fth.max()
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.035, right=0.895, top=0.90, bottom=0.08, wspace=0.07)

    titles = ["RGB–T", "Fused–RGB", "Fused–T"]
    panel_tags = ["(a)", "(b)", "(c)"]
    ims = []

    for i, (ax, diff, title) in enumerate(zip(axes, [diff_rth, diff_frgb, diff_fth], titles)):
        im = ax.imshow(
            diff, cmap="viridis", vmin=0.0, vmax=vmax,
            interpolation="nearest", rasterized=True
        )
        ims.append(im)
        style_image_axis(ax, title=title, panel_tag=panel_tags[i])

    cax = fig.add_axes([0.905, 0.16, 0.013, 0.68])
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.set_label("Abs. Difference", labelpad=6)
    cbar.ax.tick_params(labelsize=9, width=0.8, length=3)
    cbar.outline.set_linewidth(0.75)

    save_figure(fig, os.path.join(save_dir, "fig2_spectrum_difference"))
    plt.close(fig)


############################################
# Figure 3: Spectral distance bar chart
############################################
def visualize_spectral_distance(rgb_amp, th_amp, fused_amp, save_dir):
    rgb = get_shifted_amplitude_spectrum(rgb_amp, use_log=True)
    th = get_shifted_amplitude_spectrum(th_amp, use_log=True)
    fused = get_shifted_amplitude_spectrum(fused_amp, use_log=True)

    values = [
        spectral_distance(rgb, th, mode="l1"),
        spectral_distance(rgb, fused, mode="l1"),
        spectral_distance(th, fused, mode="l1"),
    ]
    labels = ["RGB–T", "RGB–F", "T–F"]

    bar_colors = [
        PAPER_COLORS["neutral"],
        PAPER_COLORS["fused"],
        PAPER_COLORS["neutral"],
    ]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(
        labels,
        values,
        color=bar_colors,
        width=0.52,
        edgecolor="#4A4A4A",
        linewidth=0.8,
        alpha=0.96
    )

    ax.set_ylabel("Mean Absolute Spectral Distance", labelpad=8)
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11)

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(values) * 0.014,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold"
        )

    ax.set_ylim(0, max(values) * 1.14)

    save_figure(fig, os.path.join(save_dir, "fig3_spectral_distance"))
    plt.close(fig)


############################################
# Appendix: Radial frequency power profile
############################################
def visualize_radial_curve(rgb_amp, th_amp, fused_amp, save_dir):
    rgb_power = get_shifted_power_spectrum(rgb_amp, use_log=False)
    th_power = get_shifted_power_spectrum(th_amp, use_log=False)
    fused_power = get_shifted_power_spectrum(fused_amp, use_log=False)

    r_rgb, p_rgb = radial_profile(rgb_power)
    r_th, p_th = radial_profile(th_power)
    r_fused, p_fused = radial_profile(fused_power)

    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(r_rgb, p_rgb, label="RGB", color=PAPER_COLORS["rgb"], alpha=0.78, linewidth=2.0)
    ax.plot(r_th, p_th, label="Thermal", color=PAPER_COLORS["thermal"], alpha=0.78, linewidth=2.0)
    ax.plot(r_fused, p_fused, label="Fused", color=PAPER_COLORS["fused"], alpha=1.0, linewidth=2.8, zorder=5)

    ax.set_yscale("log")
    ax.set_xlabel("Normalized Frequency Radius", labelpad=6)
    ax.set_ylabel("Radial Mean Power", labelpad=6)
    ax.legend(frameon=False, loc="upper right", handlelength=2.4)

    ax.grid(True, which="major", alpha=0.16, linestyle="-", color="#888888")
    ax.grid(True, which="minor", alpha=0.05, linestyle="--", color="#AAAAAA")

    save_figure(fig, os.path.join(save_dir, "supp_fig1_radial_profile"))
    plt.close(fig)


############################################
# Main entry
############################################
def run_paper_visualizations(debug, save_dir="paper_vis", include_appendix=True):
    """
    Required keys in debug:
        amp_rgb
        amp_th
        amp_fused
    """
    ensure_dir(save_dir)

    with torch.no_grad():
        visualize_spectrum(debug["amp_rgb"], debug["amp_th"], debug["amp_fused"], save_dir)
        visualize_spectrum_difference(debug["amp_rgb"], debug["amp_th"], debug["amp_fused"], save_dir)
        visualize_spectral_distance(debug["amp_rgb"], debug["amp_th"], debug["amp_fused"], save_dir)

        if include_appendix:
            visualize_radial_curve(debug["amp_rgb"], debug["amp_th"], debug["amp_fused"], save_dir)

    print("Paper visualizations saved to:", save_dir)
