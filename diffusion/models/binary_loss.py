import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryWeightedGradientLoss(nn.Module):
    """
    边界加权梯度损失（辅助训练 VAE Encoder 的 latent，使其保留显著性边界信息）
    Inputs:
        x_src: [B,C,H,W] 原图
        x_pred: [B,C,H,W] 重构图像
        mask_gt: [B,1,H,W] 二值 mask (0/1)
    Returns:
        scalar loss
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        # Sobel 核注册 buffer
        sobel_x = torch.tensor([[[-1.,0.,1.],
                                 [-2.,0.,2.],
                                 [-1.,0.,1.]]])
        sobel_y = torch.tensor([[[-1.,-2.,-1.],
                                 [0.,0.,0.],
                                 [1.,2.,1.]]])
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0))  # (1,1,3,3)
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0))

        # Dilation / Erosion 核
        self.register_buffer("morph_kernel", torch.ones(1,1,kernel_size,kernel_size))

    def _grad_mag(self, x):
        # 灰度化
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        device, dtype = x.device, x.dtype
        kx = self.sobel_x.to(device=device, dtype=dtype)
        ky = self.sobel_y.to(device=device, dtype=dtype)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, x_src: torch.Tensor, x_pred: torch.Tensor, mask_gt: torch.Tensor):
        device, dtype = x_src.device, x_src.dtype

        # ----------------- 边界权重 -----------------
        kernel = self.morph_kernel.to(device=device, dtype=dtype)
        dilated = F.conv2d(mask_gt, kernel, padding=self.kernel_size//2)
        dilated = (dilated > 0).float()
        eroded = F.conv2d(mask_gt, kernel, padding=self.kernel_size//2)
        eroded = (eroded >= kernel.sum()).float()  # 形态学侵蚀
        W_boundary = (dilated - eroded).clamp(0,1)

        # ----------------- Sobel 梯度 -----------------
        grad_src = self._grad_mag(x_src)
        grad_pred = self._grad_mag(x_pred)

        # ----------------- 边界加权 L1 -----------------
        loss = torch.sum(torch.abs(grad_pred - grad_src) * W_boundary) / (torch.sum(W_boundary) + 1e-6)
        return loss
