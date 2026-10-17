"""SpecDiff one-step latent completion with R-ASE conditioning.

Adapted from the working SpecDiff research implementation. State-dict keys
are kept stable so existing model-only checkpoints remain loadable.
"""

import torch
from torch import nn
from diffusers import DDPMScheduler
from peft import LoraConfig
from diffusion.models.autoencoder_kl import AutoencoderKL
from diffusion.models.unet_2d_condition import UNet2DConditionModel
from diffusion.models.decoder import FusionSegHead_Optimized as SimpleSegHead
from diffusion.models.rase import ResNet_Fuse_L2 as StrongCondResNet


class LatentTokenAdapter(nn.Module):
    """Map a 4-channel latent map to the shared [B, 64, 1024] token interface."""

    def __init__(self, in_channels=4, token_dim=1024, pool_size=(8, 8)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.norm = nn.LayerNorm(in_channels)
        self.proj = nn.Linear(in_channels, token_dim)

    def forward(self, latent):
        pooled = self.pool(latent)
        b, c, h, w = pooled.shape
        tokens = pooled.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return self.proj(self.norm(tokens))


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if (
        isinstance(state, dict)
        and state
        and all(str(k).startswith("module.") for k in state)
    ):
        state = {str(k)[7:]: v for k, v in state.items()}
    return state


def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        ignore_mismatched_sizes=False,
        low_cpu_mem_usage=False,
    )
    vae.requires_grad_(False)
    vae.train()

    l_target_modules_encoder = []
    l_target_modules_decoder = []
    l_grep = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
            elif ("quant_conv" in n) and ("post_quant_conv" not in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))

    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("decoder" in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
            elif ("quant_conv" in n) and ("post_quant_conv" not in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))

    lora_conf_encoder = LoraConfig(
        r=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_encoder,
    )
    lora_conf_decoder = LoraConfig(
        r=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_decoder,
    )

    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    vae.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    return vae, l_target_modules_encoder, l_target_modules_decoder


def initialize_unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv_out",
        "proj_out",
        "proj_in",
        "ff.net.2",
        "ff.net.0.proj",
        "downsamplers.0.conv",
        "upsamplers.0.conv",
        "Head",
        "Resnet",
    ]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    lora_conf_encoder = LoraConfig(
        r=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_encoder,
    )
    lora_conf_decoder = LoraConfig(
        r=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_decoder,
    )
    lora_conf_others = LoraConfig(
        r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_modules_others
    )
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


class SpecDiff(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.construction_device = torch.device(
            getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        # The research implementation enabled TF32 through a utility import.
        # Make this explicit so checkpoint inference retains its numeric setting.
        self.allow_tf32 = bool(getattr(args, "allow_tf32", True))
        if self.construction_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model, subfolder="scheduler"
        )
        self.noise_scheduler.set_timesteps(1, device=self.construction_device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            self.construction_device
        )
        self.args = args

        self.vae, self.lora_vae_modules_encoder, self.lora_vae_modules_decoder = (
            initialize_vae(self.args)
        )
        (
            self.unet,
            self.lora_unet_modules_encoder,
            self.lora_unet_modules_decoder,
            self.lora_unet_others,
        ) = initialize_unet(self.args)
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank

        self.condition_mode = getattr(args, "condition_mode", "d3")
        if self.condition_mode not in {"d0", "d1", "d2", "d3"}:
            raise ValueError(f"Unknown --condition-mode: {self.condition_mode}")
        self.train_rase_stage1 = bool(getattr(args, "train_rase_stage1", False))
        self.train_rase_decoder_only = bool(
            getattr(args, "train_rase_decoder_only", False)
        )
        self.train_seghead_only = bool(getattr(args, "train_seghead_only", False))
        self.enable_multiscale_refine = bool(
            getattr(args, "enable_multiscale_refine", False)
        )
        self.train_token_proj = bool(getattr(args, "train_token_proj", False))
        self.freeze_rase_bn = bool(getattr(args, "freeze_rase_bn", True))
        # The no-deformable Stage-I ablation uses the same parameter layout;
        # this flag controls whether the learned offset sampler is executed.
        self.disable_deformable_alignment = bool(
            getattr(args, "disable_deformable_alignment", False)
        )
        if self.train_token_proj and self.condition_mode not in {"d1", "d3"}:
            raise ValueError("--train-token-proj is only valid for d1/d3")
        if self.train_rase_stage1 and self.condition_mode != "d3":
            raise ValueError(
                "--train-rase-stage1 is only valid with --condition-mode=d3"
            )

        self.vae_thermal = None
        self.lora_vae_thermal_modules_encoder = []
        self.lora_vae_thermal_modules_decoder = []
        if self.condition_mode == "d2":
            # Native frozen VAE: no Thermal-VAE LoRA is trained in D2.
            self.vae_thermal = AutoencoderKL.from_pretrained(
                args.pretrained_model, subfolder="vae"
            )
            self.vae_thermal.requires_grad_(False)
            self.vae_thermal.eval().to(self.construction_device)

        self.unet.to(self.construction_device)
        self.vae.to(self.construction_device)
        timestep = int(getattr(args, "timestep", 499))
        if not 0 <= timestep < self.noise_scheduler.config.num_train_timesteps:
            raise ValueError("timestep is outside the pretrained scheduler range")
        self.register_buffer(
            "timesteps",
            torch.tensor([timestep], device=self.construction_device, dtype=torch.long),
            persistent=False,
        )

        rase_path = getattr(args, "rase_checkpoint", None)
        rase_state = getattr(args, "rase_state", None)
        if rase_state is None and rase_path:
            rase_state = torch.load(rase_path, map_location="cpu", weights_only=False)
        if rase_state is None:
            raise ValueError("Provide an R-ASE checkpoint or an embedded proj state")
        rase_state = _unwrap_state_dict(rase_state)
        use_quality = any(str(k).startswith("rgb_quality.") for k in rase_state)
        self.proj = StrongCondResNet(
            pretrained_backbone=False,
            use_rgb_quality=use_quality,
            rgb_quality_delta_max=1.0,
            disable_deformable_alignment=self.disable_deformable_alignment,
        ).to(self.construction_device)
        self.proj.fusion.uncertainty_freq_scale = float(
            getattr(args, "uncertainty_freq_scale", 2.0)
        )
        self.proj.fusion.phase_gate_form = str(
            getattr(args, "phase_gate_form", "bounded_log_ratio")
        )
        if self.proj.fusion.uncertainty_freq_scale < 0:
            raise ValueError("uncertainty_freq_scale must be nonnegative")
        if self.proj.fusion.phase_gate_form not in {
            "equal",
            "bounded_difference",
            "bounded_log_ratio",
        }:
            raise ValueError("Unknown phase_gate_form")
        self.proj.load_state_dict(rase_state, strict=True)
        self._set_prior_train_state()

        self.condition_adapter = None
        if self.condition_mode in {"d0", "d2"}:
            self.condition_adapter = LatentTokenAdapter(
                in_channels=4,
                token_dim=self.proj.embed_proj.out_features,
                pool_size=self.proj.pool_size,
            ).to(self.construction_device)

        self.seghead = SimpleSegHead(
            enable_multiscale_refine=self.enable_multiscale_refine
        ).to(self.construction_device)
        self.seghead.requires_grad_(True)

    def _set_prior_train_state(self):
        self.proj.requires_grad_(self.train_rase_stage1)
        self.proj.train(self.train_rase_stage1)
        if self.train_token_proj:
            self.proj.embed_proj.requires_grad_(True)
        if self.freeze_rase_bn:
            for module in self.proj.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def set_train(self):
        if self.train_seghead_only:
            self.unet.eval().requires_grad_(False)
            self.vae.eval().requires_grad_(False)
            self.proj.eval().requires_grad_(False)
            self.seghead.train().requires_grad_(True)
            if self.condition_adapter is not None:
                self.condition_adapter.eval().requires_grad_(False)
            if self.vae_thermal is not None:
                self.vae_thermal.eval().requires_grad_(False)
            return
        if self.train_rase_decoder_only:
            # Freeze diffusion weights, but keep its forward graph so the
            # segmentation loss can reach R-ASE through condition tokens.
            self.unet.eval().requires_grad_(False)
            self.vae.eval().requires_grad_(False)
            self.seghead.train().requires_grad_(True)
            self._set_prior_train_state()
            if self.vae_thermal is not None:
                self.vae_thermal.eval()
            return
        self.unet.train()
        self.vae.train()
        self.seghead.train()
        self._set_prior_train_state()
        if self.condition_adapter is not None:
            self.condition_adapter.train()
        if self.vae_thermal is not None:
            self.vae_thermal.eval()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = ".default_encoder." in n

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.seghead.eval()
        self.proj.eval()
        self.proj.requires_grad_(False)
        if self.condition_adapter is not None:
            self.condition_adapter.eval()
        if self.vae_thermal is not None:
            self.vae_thermal.eval()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = False
        self.unet.conv_in.requires_grad_(False)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = False

    def _condition_tokens(self, rgb, thermal, encoded_rgb_latent):
        zero = torch.zeros((), device=rgb.device, dtype=encoded_rgb_latent.dtype)
        empty_monitor = {"w_rgb_mean": 0.0, "w_th_mean": 0.0, "gate_mean": 0.0}
        if self.condition_mode == "d0":
            return (
                self.condition_adapter(encoded_rgb_latent),
                empty_monitor,
                None,
                zero,
                zero,
            )
        if self.condition_mode == "d1":
            with torch.set_grad_enabled(
                torch.is_grad_enabled() and self.train_token_proj
            ):
                raw_thermal = self.proj.forward_th_early(thermal)
                tokens = self.proj.feature_to_tokens(raw_thermal)
            return tokens, empty_monitor, None, zero, zero
        if self.condition_mode == "d2":
            thermal_3ch = thermal.repeat(1, 3, 1, 1)
            with torch.no_grad():
                thermal_latent = self.vae_thermal.encode(
                    thermal_3ch
                ).latent_dist.sample()
                thermal_latent = thermal_latent * self.vae_thermal.config.scaling_factor
            return (
                self.condition_adapter(thermal_latent),
                empty_monitor,
                None,
                zero,
                zero,
            )
        if self.train_rase_stage1 or self.train_token_proj:
            # Frozen backbone parameters and image inputs require no graph;
            # only embed_proj creates one when token-only training is enabled.
            tokens, monitor, mid_mask, align_loss, spatial_loss, _ = self.proj(
                rgb, thermal, return_diagnostics=False
            )
        else:
            with torch.no_grad():
                tokens, monitor, mid_mask, align_loss, spatial_loss, _ = self.proj(
                    rgb, thermal, return_diagnostics=False
                )
        return tokens, monitor, mid_mask, align_loss, spatial_loss

    def forward(self, c_t, visual_embedding):

        encoded_control = (
            self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        )  # b, 4, 16, 16

        visual_embeds, monitor, mid_mask, align_loss, loss_align_spatial = (
            self._condition_tokens(c_t, visual_embedding, encoded_control)
        )

        model_pred_all = self.unet(
            encoded_control,
            self.timesteps,
            encoder_hidden_states=visual_embeds.to(torch.float32),
        )
        model_pred = model_pred_all.sample

        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            encoded_control.device
        )
        x_denoised = self.noise_scheduler.step(
            model_pred, self.timesteps, encoded_control, return_dict=True
        ).prev_sample

        logits = self.seghead(x_denoised, model_pred_all.cross_attn_feats)

        return (
            logits,
            x_denoised,
            mid_mask,
            monitor,
            align_loss,
            visual_embeds,
            loss_align_spatial,
        )

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["vae_lora_decoder_modules"] = self.lora_vae_modules_decoder

        sd["vae_thermal_lora_encoder_modules"] = self.lora_vae_thermal_modules_encoder
        sd["vae_thermal_lora_decoder_modules"] = self.lora_vae_thermal_modules_decoder

        (
            sd["unet_lora_encoder_modules"],
            sd["unet_lora_decoder_modules"],
            sd["unet_lora_others_modules"],
        ) = (
            self.lora_unet_modules_encoder,
            self.lora_unet_modules_decoder,
            self.lora_unet_others,
        )
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {
            k: v
            for k, v in self.unet.state_dict().items()
            if "lora" in k or "conv_in" in k
        }
        sd["state_dict_vae"] = {
            k: v for k, v in self.vae.state_dict().items() if "lora" in k
        }
        sd["state_dict_vae_thermal"] = (
            {k: v for k, v in self.vae_thermal.state_dict().items() if "lora" in k}
            if self.vae_thermal is not None
            else {}
        )
        sd["condition_mode"] = self.condition_mode
        sd["train_rase_stage1"] = self.train_rase_stage1
        sd["train_rase_decoder_only"] = self.train_rase_decoder_only
        sd["train_seghead_only"] = self.train_seghead_only
        sd["enable_multiscale_refine"] = self.enable_multiscale_refine
        sd["train_token_proj"] = self.train_token_proj
        sd["freeze_rase_bn"] = self.freeze_rase_bn
        sd["training_protocol_version"] = 4
        sd["model_name"] = "SpecDiff"
        sd["timestep"] = int(self.timesteps.item())
        sd["disable_deformable_alignment"] = self.disable_deformable_alignment
        sd["allow_tf32"] = self.allow_tf32
        sd["uncertainty_freq_scale"] = self.proj.fusion.uncertainty_freq_scale
        sd["phase_gate_form"] = self.proj.fusion.phase_gate_form
        sd["rase_checkpoint"] = (
            None  # weights are embedded; no machine path is serialized
        )
        sd["proj"] = self.proj.state_dict()
        if self.condition_adapter is not None:
            sd["condition_adapter"] = self.condition_adapter.state_dict()
        sd["seghead"] = self.seghead.state_dict()

        torch.save(sd, outf)
