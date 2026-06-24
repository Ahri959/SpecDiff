import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'diffusion'))

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
from my_utils.vaehook import VAEHook
from torchvision import models


import torch.nn.functional as F
from timm import create_model
import torchvision.models as models
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from transformers import AutoImageProcessor, AutoModelForImageClassification
from diffusion.models.ResUnet_seghead import FusionSegHead_Optimized as SimpleSegHead
from diffusion.models.Swin import SwinBackbone

from diffusion.models.resnet_model import ResNet_Fuse_L2 as StrongCondResNet
from diffusion.models.FFP_complex_module import UltimateFrequencyFusionV3 as UltimateFrequencyFusion

def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(
    args.pretrained_model,
    subfolder="vae",
    ignore_mismatched_sizes=True,  # 忽略不匹配的 shape
    low_cpu_mem_usage=False        # 必须关闭，否则不会 overwrite
)
    vae.requires_grad_(False)
    vae.train()

    l_target_modules_encoder = []
    l_target_modules_decoder = []
    l_grep = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("decoder" in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
                
    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_decoder)
    
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    vae.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    return vae, l_target_modules_encoder,l_target_modules_decoder


def initialize_unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out",
              "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj", "downsamplers.0.conv", "upsamplers.0.conv","Head","Resnet" ]
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

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


class specdiff_train(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        self.vae, self.lora_vae_modules_encoder, self.lora_vae_modules_decoder = initialize_vae(self.args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(
            self.args)
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank
        
        self.vae_thermal, self.lora_vae_thermal_modules_encoder, self.lora_vae_thermal_modules_decoder = initialize_vae(self.args)

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.vae_thermal.to("cuda")
        self.timesteps = torch.tensor([499], device="cuda").long()
    

 
        
        self.proj = StrongCondResNet()
        self.proj.load_state_dict(torch.load("/home/user/specdiff/pre_model/align_agnostic_train.pth", map_location="cuda"))  # 加载预训练模型
        self.proj.requires_grad_(True)
        
        self.seghead = SimpleSegHead().to("cuda")
        self.seghead.requires_grad_(True)
        
  
        
        
    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = False
        self.unet.conv_in.requires_grad_(False)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = False

    @torch.no_grad()
    def eval(self, lq, visual_embedding):
        lq_latent = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(lq_latent, self.timesteps,
                               encoder_hidden_states=self.proj(visual_embedding.to(torch.float32))).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = self.seghead(output_image)
        return output_image

    def forward(self, c_t, visual_embedding):
        
        
        
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor  # b, 4, 16, 16
       
        
        visual_embeds, monitor, mid_mask, align_loss, loss_align_spatial= self.proj(c_t, visual_embedding)
        

        model_pred_all = self.unet(encoded_control, self.timesteps,
                               encoder_hidden_states=visual_embeds.to(torch.float32), )
        model_pred = model_pred_all.sample
        
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control,
                                               return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        
        logits = self.seghead(x_denoised, model_pred_all.cross_attn_feats)
        
        return logits, x_denoised, mid_mask, monitor, align_loss,visual_embeds,loss_align_spatial

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["vae_lora_decoder_modules"] = self.lora_vae_modules_decoder
        
        sd["vae_thermal_lora_encoder_modules"] = self.lora_vae_thermal_modules_encoder
        sd["vae_thermal_lora_decoder_modules"] = self.lora_vae_thermal_modules_decoder
        
        
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] = \
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["state_dict_vae_thermal"] = {k: v for k, v in self.vae_thermal.state_dict().items() if "lora" in k}
        
        
        sd["proj"] = self.proj.state_dict()
        sd["seghead"] = self.seghead.state_dict()
      

               
        torch.save(sd, outf)


class specdiff_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model, subfolder="unet")

        self.vae_thermal= AutoencoderKL.from_pretrained(self.args.pretrained_model, subfolder="vae")


        self.proj = StrongCondResNet().to("cuda")
        self.proj.requires_grad_(True)
        self.seghead = SimpleSegHead().to("cuda")
        self.seghead.requires_grad_(True)
        

        # vae tile
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size,
                             decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        if args.specdiff_path is None:
            print('Do not provide any specdiff path!')
        else:
            specdiff = torch.load(args.specdiff_path)
            self.load_ckpt(specdiff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([499], device="cuda").long()
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

    def load_ckpt(self, model):
        # load unet lora
        #self.proj.load_state_dict(model["proj"])
        ckpt = torch.load("/home/user/specdiff/pre_model/align_agnostic_train.pth")
        self.proj.load_state_dict(ckpt)
        self.seghead.load_state_dict(model["seghead"])

        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                      target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian",
                                           target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        
        vae_lora_conf_decoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian",
                                           target_modules=model["vae_lora_decoder_modules"])
        self.vae.add_adapter(vae_lora_conf_decoder, adapter_name="default_decoder")
        
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder', 'default_decoder'])
            
        # 重建 LoRA 配置
        vae_t_lora_conf_encoder = LoraConfig(
            r=model["rank_vae"],
            init_lora_weights="gaussian",
            target_modules=model["vae_thermal_lora_encoder_modules"]
        )
        vae_t_lora_conf_decoder = LoraConfig(
            r=model["rank_vae"],
            init_lora_weights="gaussian",
            target_modules=model["vae_thermal_lora_decoder_modules"]
        )

        # 附加到模型
        self.vae_thermal.add_adapter(vae_t_lora_conf_encoder, adapter_name="default_encoder")
        self.vae_thermal.add_adapter(vae_t_lora_conf_decoder, adapter_name="default_decoder")

        # 加载参数
        for n, p in self.vae_thermal.named_parameters():
            if "lora" in n and n in model["state_dict_vae_thermal"]:
                p.data.copy_(model["state_dict_vae_thermal"][n])

        # 启用 thermal LORA
        self.vae_thermal.set_adapter(["default_encoder", "default_decoder"])
    
    # @perfcount
 
    def forward(self, lq, visual_embedding, gt_mask=None):
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
   
        
        visual_embeds, monitor, mid_mask, _, _= self.proj(lq, visual_embedding)


        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            model_pred_all = self.unet(lq_latent, self.timesteps, encoder_hidden_states=visual_embeds.float())
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols - 1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        model_out = self.unet(input_list_t, self.timesteps,
                                              encoder_hidden_states=visual_embedding).sample
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[
                                                                                                  row * grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.noise_scheduler.step(model_pred_all.sample, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (
            self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_logits = self.seghead(x_denoised, model_pred_all.cross_attn_feats)#, model_pred_all.encoder_hidden_states)
        
        
        return output_image,output_logits


    def _init_tiled_vae(self,
                        encoder_tile_size=256,
                        decoder_tile_size=256,
                        fast_decoder=False,
                        fast_encoder=False,
                        color_fix=False,
                        vae_to_gpu=True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder,
            color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder,
            color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var) for x
            in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [
            exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var) for
            y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))
