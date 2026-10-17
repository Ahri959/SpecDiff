import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# ------------------ 工具函数 ------------------
def _to_prob(pred):
    """保证预测为概率 [0,1]"""
    if pred.max() > 1 or pred.min() < 0:
        return torch.sigmoid(pred)
    return pred

def _centroid(gt):
    """计算前景质心"""
    rows, cols = gt.shape[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = round(cols / 2)
        Y = round(rows / 2)
    else:
        total = gt.sum()
        i = torch.arange(0, cols).cuda().float()
        j = torch.arange(0, rows).cuda().float()
        X = torch.round((gt.sum(dim=0)*i).sum() / total).item()
        Y = torch.round((gt.sum(dim=1)*j).sum() / total).item()
    X = max(1, int(X))
    Y = max(1, int(Y))
    return X, Y

def _divideGT(gt, X, Y):
    h, w = gt.shape[-2:]
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:]
    LB = gt[Y:, :X]
    RB = gt[Y:, X:]
    area = h * w
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def _dividePrediction(pred, X, Y):
    h, w = pred.shape[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:]
    LB = pred[Y:, :X]
    RB = pred[Y:, X:]
    return LT, RT, LB, RB

def _ssim(pred, gt):
    """简单 SSIM，用于 S-measure region 部分"""
    pred = pred.float()
    gt = gt.float()
    if pred.numel() == 0 or gt.numel() == 0:
        return 0.0
    N = pred.numel()
    if N <= 1:
        return 0.0
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x)**2).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y)**2).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)

    alpha = 4 * x * y * sigma_xy
    beta = (x*x + y*y) * (sigma_x2 + sigma_y2)
    if alpha != 0:
        Q = alpha / (beta + 1e-20)
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0.0
    return Q

def _object(pred, gt):
    """S-measure object 部分"""
    temp = pred[gt == 1]
    if temp.numel() == 0:
        return 0.0
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x*x + 1.0 + sigma_x + 1e-20)
    return score

def _S_object(pred, gt):
    fg = torch.where(gt==0, torch.zeros_like(pred), pred)
    bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1-gt)
    u = gt.mean()
    Q = u * o_fg + (1-u) * o_bg
    return Q

def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
    return Q

# ------------------ S-measure ------------------
def compute_sm(pred, gt, alpha=0.5, device='cuda'):
    """
    计算单张图像的 S-measure
    pred: [H,W] 或 PIL Image 或 Tensor
    gt: [H,W] 或 PIL Image 或 Tensor
    """
    # 转为 Tensor
    if not torch.is_tensor(pred):
        pred = transforms.ToTensor()(pred)
    if not torch.is_tensor(gt):
        gt = transforms.ToTensor()(gt)

    pred = pred.to(device).float()
    gt = gt.to(device).float()
    pred = _to_prob(pred)

    y = gt.mean()
    if y == 0:
        Q = 1.0 - pred.mean()
    elif y == 1:
        Q = pred.mean()
    else:
        Q = alpha * _S_object(pred, gt) + (1-alpha) * _S_region(pred, gt)
        if torch.isnan(Q) or Q < 0:
            Q = torch.tensor(0.0, device=device)

    return float(Q)


# ------------------- 统一处理 logits / prob -------------------
def _to_prob(pred):
    """
    Convert logits to probability [0,1].
    If already in [0,1], keep unchanged.
    """
    if pred.max() > 1 or pred.min() < 0:
        return torch.sigmoid(pred)
    return pred

# ------------------- IoU -------------------
def compute_iou(pred, gt, threshold=0.5):
    pred = _to_prob(pred)
    gt = gt.to(pred.device)
    pred_bin = (pred > threshold).float()

    intersection = (pred_bin * gt).sum(dim=[1,2,3])
    union = (pred_bin + gt - pred_bin*gt).sum(dim=[1,2,3])
    iou_per_image = intersection / (union + 1e-6)
    return iou_per_image.mean().item()




# ------------------- MAE -------------------
def compute_mae(pred, gt):
    pred = _to_prob(pred)
    gt = gt.to(pred.device)
    mae_per_image = torch.mean(torch.abs(pred - gt), dim=[1,2,3])
    return mae_per_image.mean().item()

# ------------------- Enhanced-alignment measure -------------------
def compute_em(pred, gt):
    pred = _to_prob(pred)
    gt = (gt > 0.5).float()

    B = pred.shape[0]
    Em_list = []

    for i in range(B):
        p = pred[i,0]
        g = gt[i,0]

        mean_p = p.mean()
        mean_g = g.mean()
        p_t = p - mean_p
        g_t = g - mean_g

        align_matrix = 2 * (p_t * g_t) / (p_t*p_t + g_t*g_t + 1e-8)
        enhanced = ((align_matrix + 1)**2)/4
        Em_list.append(enhanced.mean())

    return torch.stack(Em_list).mean().item()




# ------------------- Max F-measure -------------------
def compute_fmax(pred, gt, beta2=0.3):
    pred = _to_prob(pred)
    gt = (gt > 0.5).float()

    B = pred.shape[0]
    fmax_list = []

    for i in range(B):
        p = pred[i,0]
        g = gt[i,0]

        thresholds = torch.linspace(0,1,255, device=p.device)
        f_scores = []

        for t in thresholds:
            pb = (p >= t).float()
            TP = (pb * g).sum()
            precision = TP / (pb.sum() + 1e-6)
            recall = TP / (g.sum() + 1e-6)
            f = (1 + beta2) * precision * recall / (beta2*precision + recall + 1e-6)
            f_scores.append(f)

        f_scores = torch.stack(f_scores)
        f_scores[f_scores != f_scores] = 0
        fmax_list.append(f_scores.max())

    return torch.stack(fmax_list).mean().item()

# ------------------- 统一指标计算 -------------------
def compute_metrics(pred, gt):
    """
    Compute IoU, MAE, Max F-measure, S-measure, E-measure
    Supports logits or probability inputs.
    pred: [B,1,H,W]
    gt  : [B,1,H,W]
    """
    iou = compute_iou(pred, gt)
    mae = compute_mae(pred, gt)
    fmax = compute_fmax(pred, gt)
    sm = compute_sm(pred, gt)
    em = compute_em(pred, gt)

    return {
        'IoU': iou,
        'MAE': mae,
        'F-measure': fmax,
        'S-measure': sm,
        'E-measure': em
    }
