import os
import os.path
import argparse
import torch
import torch.nn.functional as F

import sys

import torch.utils.checkpoint
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
import pyiqa
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from diffusion.SpecDiff import SpecDiff_train
from diffusion.models.discriminator import Discriminator

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from data.dataset_jpeg import DatasetJPEG
from utils import utils_image as util
from utils import utils_option as option

from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import warnings

warnings.filterwarnings("ignore")

import wandb
from datetime import datetime
import torchvision
from diffusion.models.binary_loss import BoundaryWeightedGradientLoss as BoundaryWeightedGradientLoss

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

import utils.utils_image as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils.utils_metric import compute_em, compute_iou, compute_mae


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")


def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")


def parse_str_list(arg):
    return arg.split(',')

def dice_loss_helper(pred_prob, target, smooth=1e-5):
    """
    辅助函数：计算 Dice Loss
    注意：这个函数期望的输入是 概率 (probabilities)，即 sigmoid 激活后的值。
    
    :param pred_prob: 模型的概率输出 (已 sigmoid) (B, C, H, W)
    :param target: 真实掩码 (B, C, H, W)
    :param smooth: 防止除以零的平滑值
    """
    # 确保 target 是 float 类型
    target = target.float()
    
    # 展平 tensor
    pred_flat = pred_prob.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # 计算交集 (intersection)
    intersection = (pred_flat * target_flat).sum()
    
    # 计算 Dice 系数 (coefficient)
    dice_coefficient = (2. * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    
    # 返回 Dice Loss (1 - Dice 系数)
    return 1 - dice_coefficient


# def structure_loss(pred, mask, bce_weight=1.0, dice_weight=1.0):
#     """
#     【修改后】
#     这是 BCE + Dice Loss 的稳定组合。
    
#     重要：此函数假定 'pred' 是模型的原始 LOGITS (未经 sigmoid 激活)。
#     'mask' 是 0/1 的真实掩码。
    
#     :param pred: 模型的 Logits 输出 (B, C, H, W)
#     :param mask: 真实掩码 (B, C, H, W)
#     :param bce_weight: BCE 损失的权重
#     :param dice_weight: Dice 损失的权重
#     """
    
#     # --- 1. 计算 BCE Loss ---
#     # F.binary_cross_entropy_with_logits 会自动、稳定地处理 Logits
#     loss_bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    
#     # --- 2. 计算 Dice Loss ---
#     # (a) Dice Loss 需要概率，所以我们在这里对 Logits 应用 sigmoid
#     pred_prob = torch.sigmoid(pred)
    
#     # (b) 使用辅助函数计算 Dice Loss
#     loss_dice = dice_loss_helper(pred_prob, mask)
    
#     # --- 3. 组合损失 ---
#     # 返回加权后的总损失。1:1 的权重是一个很好的起点。
#     return (bce_weight * loss_bce) + (dice_weight * loss_dice)




def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()




def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    # training details
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--max_train_steps", type=int, default=1000000, ) # 100000
    parser.add_argument("--checkpointing_steps", type=int, default=5000, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--learning_rate_re", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0, )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument("--tracker_project_name", type=str, default="train_SpecDiff",
                        help="The name of the wandb project to log to.")
    parser.add_argument("--pretrained_model", default=None, type=str)
    parser.add_argument('--gan_dis_weight', default=1e-2, type=float)
    parser.add_argument('--gan_gen_weight', default=5e-3, type=float)

    # lora setting
    parser.add_argument("--lora_rank", default=4, type=int)

    # dataset setting
    parser.add_argument("--datasets", default='options/SpecDiff.json')

    # other setting
    parser.add_argument('--cave_path', type=str, required=True)
    parser.add_argument('--val_path', required=True)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument('--debug', action='store_true')


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args



from peft import LoraConfig

def load_ckpt(model_gen, model):
    """
    只加载已有模型参数到 model_gen，
    不创建 LoRA adapter、不使用 default_encoder/default_decoder。
    """
    # 1. 加载 projection 层参数
    if "proj" in model:
        model_gen.proj.load_state_dict(model["proj"])
    
    # 2. 加载 segmentation head 参数
    if "seghead" in model:
        model_gen.seghead.load_state_dict(model["seghead"])

    # 3. 加载 UNet 参数（仅复制已存在参数）
    if "state_dict_unet" in model:
        for n, p in model_gen.unet.named_parameters():
            if n in model["state_dict_unet"]:
                p.data.copy_(model["state_dict_unet"][n])

    # 4. 加载 VAE 参数（仅复制已存在参数）
    if "state_dict_vae" in model:
        for n, p in model_gen.vae.named_parameters():
            if n in model["state_dict_vae"]:
                p.data.copy_(model["state_dict_vae"][n])

    print("✅ 模型参数加载完成（未启用 LoRA adapter）")
    return model_gen



def main(args):
    args.tracker_project_name = os.path.join("training_results", args.tracker_project_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    logging_dir = Path(args.tracker_project_name, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.tracker_project_name, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.tracker_project_name, "checkpoints"), exist_ok=True)
        if not args.debug:
            wandb.init(project="diff-car", name=args.tracker_project_name)

    model_gen = SpecDiff_train(args)
    model_gen.set_train()
    model_reg = Discriminator(args=args, accelerator=accelerator)
    model_reg.set_train()
    

    #loss_fn = pyiqa.create_metric('dists', device=accelerator.device, as_loss=True)
    #mask_perc_loss = JointModalPerceptualLoss(device='cuda')
    
    # set vae adapter
    model_gen.vae.set_adapter(['default_encoder'])
    # set gen adapter
    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    
    # SpecDiff = torch.load("training_results/train_SpecDiff/2025-11-13_19-25-37/checkpoints/model_15000.pkl")
    # model_gen = load_ckpt(model_gen,SpecDiff)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
            model_reg.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        model_reg.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = []
    for n, _p in model_gen.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    if hasattr(model_gen, 'proj'):
        layers_to_opt += list(model_gen.proj.parameters())
    if hasattr(model_gen, 'seghead'):
        layers_to_opt += list(model_gen.seghead.parameters())

        
    for n, _p in model_gen.vae.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
            
    for n, _p in model_gen.vae_thermal.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )



    layers_to_opt_reg = []
    for n, _p in model_reg.unet.named_parameters():
        if "lora" in n:
            layers_to_opt_reg.append(_p)
    for _p in model_reg.cls_pred_branch.parameters():
        layers_to_opt_reg.append(_p)
    layers_to_opt_reg.append(model_reg.embeddings)

    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
                                      betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                      eps=args.adam_epsilon, )
    lr_scheduler_reg = get_scheduler(args.lr_scheduler, optimizer=optimizer_reg,
                                     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                     num_training_steps=args.max_train_steps,
                                     num_cycles=args.lr_num_cycles, power=args.lr_power)

    args.datasets = option.parse_dataset(args.datasets)['datasets']
    for phase, dataset_opt in args.datasets.items():
        if phase == 'train':
            train_set = DatasetJPEG(dataset_opt)
            train_set.normalize = True
            dl_train = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)



    # Prepare everything with our `accelerator`.
    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
    )

    if accelerator.is_main_process:
        del args.datasets
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, total=args.max_train_steps)
    BoundaryWeightedGradientLoss

    # start the training loop
    global_step = 0
    best_mae = 1.0
    best_iou = 0
    avg_iou = 0.9
    avg_mae = 0.05
    best_em = 0
    avg_em = 0.9

    while True:
        for step, batch in enumerate(dl_train):
            global_step += 1
            if global_step > args.max_train_steps:
                exit()
            m_acc = [model_gen, model_reg]
            with accelerator.accumulate(*m_acc):
                
                
                x_src = batch["img"].to("cuda").float()
                x_tgt = batch["mask"].to("cuda").float()
                visual_embedding = batch["thermal"].to("cuda")
                
                # forward pass
                logits_pre,  latents_pred, mid_mask, monitor, align_loss ,z_cond,loss_align_spatial= model_gen(x_src, visual_embedding) # visual_embedding.shape:[1, 64, 512]
                
                

                
                mask_loss = structure_loss(logits_pre.float(),x_tgt.float())
                
                mid_mask_loss = structure_loss(mid_mask.float(),x_tgt.float())
             
                
                
                if torch.cuda.device_count() > 1:
                    generator_loss = model_reg.module.compute_generator_loss(latents_pred,z_cond)
                else:
                    generator_loss = model_reg.compute_generator_loss(latents_pred,z_cond)
                
                

                
                loss =  mask_loss + 0.5 * mid_mask_loss + 0.1 * loss_align_spatial+ generator_loss * args.gan_gen_weight
                
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)


            
                # discriminator loss
              
                if torch.cuda.device_count() > 1:
                    gt_latents = model_reg.module.compute_gt_latents(x_src)
                    loss_d = model_reg.module.compute_discriminator_loss(gt_latents, latents_pred, z_cond) * args.gan_dis_weight
                else:
                    gt_latents = model_reg.compute_gt_latents(x_src)
                    loss_d = model_reg.compute_discriminator_loss(gt_latents, latents_pred,z_cond) * args.gan_dis_weight
                
          
                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none) 

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)

                if accelerator.is_main_process:

                    logs = {}
                    # log all the losses
                    logs["loss_d"] = loss_d.detach().item()
                    logs["loss_g"] = generator_loss.detach().item()
                    logs["mask_loss"] = mask_loss.detach().item() 
                  
                    logs["avg_mae"] = avg_mae
                    logs["w_rgb_mean"] = monitor['w_rgb_mean']
                    logs["w_th_mean"] = monitor['w_th_mean']
                    logs["gate_mean"] = monitor['gate_mean']
                    progress_bar.set_postfix(**logs)
                    if not args.debug:
                        wandb.log({'loss_d': logs['loss_d'],'loss_g': logs['loss_g'],'w_rgb_mean': logs['w_rgb_mean'],'w_th_mean': logs['w_th_mean'],'gate_mean': logs['gate_mean'],'align_loss': logs['align_loss'],'mask_loss': logs['mask_loss'],'avg_mae':logs['avg_mae']}, step=global_step)

                    accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0:
                H_paths = utils.get_image_paths("/home/user/datasets/VT5000/test/images")
                

                device = 'cuda'
            


            
                #os.makedirs(os.path.join(args.output_dir, exist_ok=True))

                mae_list=[]
                iou_list=[]
                Em_list = []
                cnt = 0
                
                for idx, img in tqdm(enumerate(H_paths)):



                    img_pth = img
                    mask_path =img_pth.replace("images","masks").replace("jpg","png")
                    thermal_pth = mask_path.replace("masks","thermal").replace("png","jpg")
                    
                    mask = Image.open(mask_path).convert('L')
                    gt_img = Image.open(img_pth).convert("RGB")
                    thermal = Image.open(thermal_pth).convert("L")

                    gt_img = gt_img.resize((384,384), Image.BILINEAR)
                    thermal = thermal.resize((384,384), Image.BILINEAR)
                    mask    = mask.resize((384,384), Image.NEAREST)  # mask 禁止 bilinear！

                    gt_img = transforms.ToTensor()(gt_img)
                    thermal = transforms.ToTensor()(thermal)
                    mask = transforms.ToTensor()(mask)
                    
                    
                    gt_img = torchvision.transforms.functional.normalize(gt_img, mean=[0.5], std=[0.5])
                    thermal = torchvision.transforms.functional.normalize(thermal, mean=[0.5], std=[0.5])
                    
                    gt_img = gt_img.to('cuda').float().unsqueeze(0)
                    mask = mask.to('cuda').float().unsqueeze(0)
                    visual_embedding = thermal.to('cuda').float().unsqueeze(0)
                    
                        
                        # translate the image
                    with torch.no_grad():
                        if torch.cuda.device_count() > 1:
                            logits,  latents_pred, mid_mask,_ , align_loss,z_cond,_ = model_gen(gt_img.float(), visual_embedding.float())
                        else:
                            logits, latents_pred, mid_mask,_ , align_loss ,z_cond,_ = model_gen(gt_img.float(), visual_embedding.float())

                    
                    
                    mae = compute_mae((logits).float(),mask.float())
                    #iou = compute_iou(logits.float(),mask.float())
                    #mae = compute_mae((torch.sigmoid(logits)).float(),mask)

                    #Em_list.append(Em)
                    #iou_list.append(iou)
                    mae_list.append(mae)

                #avg_em = sum(Em_list) / len(Em_list)
                avg_mae = sum(mae_list) / len(mae_list)
                #avg_iou = sum(iou_list) / len(iou_list)

                if best_mae > avg_mae:
                    best_mae = avg_mae
                
            
                    # checkpoint the model
                    outf = os.path.join(args.tracker_project_name, "checkpoints", f"model_VT5000_{global_step}_Em_{avg_mae:.4f}.pkl")
                    accelerator.unwrap_model(model_gen).save_model(outf)
                model_gen.train()
                


if __name__ == "__main__":
    args = parse_args()
    main(args)
