"""Cross-Modal Consistency Discriminator (CMCD).

Architecture follows the latent discriminator used by Compression-Aware
One-Step Diffusion; see THIRD_PARTY_NOTICES.md for attribution. Full CMCD
adds sample-shuffled negative pairs. There is no extra compatibility head
and no pair-ranking objective in this release's paper configuration.
"""

import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig
from diffusion.models.unet_2d_condition import UNet2DConditionModel
from diffusion.models.autoencoder_kl import AutoencoderKL


class CMCD(nn.Module):
    def __init__(self, pretrained_model, lora_rank=4, mode="full"):
        super().__init__()
        if mode not in {"latent", "conditional", "full"}:
            raise ValueError(mode)
        self.mode = mode
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet"
        )
        self.unet.requires_grad_(False)
        targets = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ]
        self.unet.add_adapter(LoraConfig(r=lora_rank, target_modules=targets))
        del self.unet.up_blocks
        del self.unet.conv_out
        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
        self.vae.requires_grad_(False).eval()
        del self.vae.decoder
        self.cls_pred_branch = nn.Sequential(
            nn.Conv2d(1280, 2560, 4, 1, 1),
            nn.GroupNorm(32, 2560),
            nn.SiLU(),
            nn.Conv2d(2560, 1, 4, 1, 1),
        )
        self.embeddings = nn.Parameter(
            torch.randn(77, 1024), requires_grad=mode == "latent"
        )

    def set_train(self):
        self.train()
        self.vae.eval()
        for name, p in self.unet.named_parameters():
            p.requires_grad_("lora" in name)
        self.cls_pred_branch.requires_grad_(True)
        self.embeddings.requires_grad_(self.mode == "latent")

    def set_eval(self):
        self.eval()
        self.requires_grad_(False)

    def compute_cls_logits(self, latent, condition):
        if self.mode == "latent":
            condition = self.embeddings[None].expand(latent.shape[0], -1, -1)
        elif condition is None:
            raise ValueError("Conditional CMCD requires prior tokens")
        feature = self.unet(
            latent,
            torch.full((latent.shape[0],), 199, device=latent.device, dtype=torch.long),
            encoder_hidden_states=condition.float(),
            as_discriminator=True,
        )
        logits = self.cls_pred_branch(feature).float() / 2.5
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite CMCD logits")
        return logits

    def generator_loss(self, latent, condition):
        self.set_eval()
        score = self.compute_cls_logits(latent, condition)
        return F.binary_cross_entropy_with_logits(score, torch.ones_like(score))

    def discriminator_loss(
        self, real, completed, condition, balance_negative_weight=False
    ):
        self.set_train()
        condition = condition.detach()
        matched = self.compute_cls_logits(real.detach(), condition)
        fake = self.compute_cls_logits(completed.detach(), condition)
        loss_real = F.binary_cross_entropy_with_logits(
            matched, torch.ones_like(matched)
        )
        loss_fake = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        metrics = dict(
            d_match_prob=matched.sigmoid().mean().detach(),
            d_fake_prob=fake.sigmoid().mean().detach(),
        )
        if self.mode != "full":
            return (
                loss_real + (2 if balance_negative_weight else 1) * loss_fake,
                metrics,
            )
        batch = real.shape[0]
        if batch < 2:
            raise ValueError("Full CMCD requires at least two samples per device")
        offset = torch.randint(1, batch, (), device=real.device)
        permutation = (torch.arange(batch, device=real.device) + offset) % batch
        mismatch = self.compute_cls_logits(real.detach(), condition[permutation])
        loss_mismatch = F.binary_cross_entropy_with_logits(
            mismatch, torch.zeros_like(mismatch)
        )
        metrics["d_mismatch_prob"] = mismatch.sigmoid().mean().detach()
        return loss_real + loss_fake + loss_mismatch, metrics

    @torch.no_grad()
    def real_latent(self, rgb):
        return (
            self.vae.encode(rgb).latent_dist.sample() * self.vae.config.scaling_factor
        )
