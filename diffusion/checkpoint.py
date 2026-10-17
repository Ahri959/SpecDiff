"""Strict model-only checkpoint loading, including existing SpecDiff weights."""

from pathlib import Path
from types import SimpleNamespace
import torch


def read_checkpoint(path):
    # Checkpoints must come from a trusted source; original files contain metadata.
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_tensors(module, state, name, complete=False):
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Empty or missing checkpoint section: {name}")
    target = module.state_dict()
    bad = [k for k, v in state.items() if k not in target or target[k].shape != v.shape]
    if bad:
        raise RuntimeError(f"{name} has unknown or incompatible keys: {bad[:8]}")
    if complete and set(state) != set(target):
        raise RuntimeError(
            f"{name} is incomplete: {sorted(set(target)-set(state))[:8]}"
        )
    module.load_state_dict(state, strict=complete)
    print(f"[load] {name}: {len(state)}/{len(state)} source tensors")


def load_generator_weights(model, checkpoint, load_rase=True):
    for name, section in [("unet", "state_dict_unet"), ("vae", "state_dict_vae")]:
        module = getattr(model, name)
        state = checkpoint.get(section)
        load_tensors(module, state, section)
        required = {
            k
            for k in module.state_dict()
            if "lora" in k or (name == "unet" and k.startswith("conv_in."))
        }
        missing = required - set(state)
        if missing:
            raise RuntimeError(
                f"{section} lacks required trainable tensors: {sorted(missing)[:8]}"
            )
    load_tensors(model.seghead, checkpoint.get("seghead"), "seghead", complete=True)
    if load_rase:
        load_tensors(model.proj, checkpoint.get("proj"), "proj", complete=True)
    if model.condition_adapter is not None:
        load_tensors(
            model.condition_adapter,
            checkpoint.get("condition_adapter"),
            "condition_adapter",
            complete=True,
        )
    model.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
    model.vae.set_adapter(["default_encoder"])


def build_model(
    checkpoint,
    pretrained_model,
    rase_checkpoint=None,
    device="cuda",
    timestep=None,
    disable_deformable_alignment=None,
):
    from diffusion.specdiff import SpecDiff

    rank = int(checkpoint.get("rank_unet", 4))
    if int(checkpoint.get("rank_vae", rank)) != rank:
        raise ValueError("VAE and UNet LoRA ranks must match in this implementation")
    if rase_checkpoint is None and not checkpoint.get("proj"):
        raise ValueError("Checkpoint has no embedded R-ASE; supply --rase-checkpoint")
    args = SimpleNamespace(
        pretrained_model=str(pretrained_model),
        device=device,
        lora_rank=rank,
        rase_checkpoint=rase_checkpoint,
        rase_state=None if rase_checkpoint else checkpoint.get("proj"),
        condition_mode=checkpoint.get("condition_mode", "d3"),
        train_rase_stage1=False,
        train_token_proj=False,
        freeze_rase_bn=True,
        enable_multiscale_refine=bool(checkpoint.get("enable_multiscale_refine", False))
        or any(
            k.startswith("multiscale_refine.") for k in checkpoint.get("seghead", {})
        ),
        disable_deformable_alignment=(
            bool(checkpoint.get("disable_deformable_alignment", False))
            if disable_deformable_alignment is None
            else disable_deformable_alignment
        ),
        uncertainty_freq_scale=float(checkpoint.get("uncertainty_freq_scale", 2.0)),
        phase_gate_form=checkpoint.get("phase_gate_form", "bounded_log_ratio"),
        timestep=int(checkpoint.get("timestep", 499)) if timestep is None else timestep,
        allow_tf32=bool(checkpoint.get("allow_tf32", True)),
    )
    if rase_checkpoint:
        state = read_checkpoint(rase_checkpoint)
        if isinstance(state, dict) and isinstance(state.get("config"), dict):
            config = state["config"]
            if disable_deformable_alignment is None:
                args.disable_deformable_alignment = bool(
                    config.get("disable_deformable_alignment", False)
                )
            args.uncertainty_freq_scale = float(
                config.get("uncertainty_freq_scale", 2.0)
            )
            args.phase_gate_form = config.get("phase_gate_form", "bounded_log_ratio")
    model = SpecDiff(args).to(device)
    load_generator_weights(model, checkpoint, load_rase=rase_checkpoint is None)
    model.eval()
    model.set_eval()
    return model
