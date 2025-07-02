import torch
import torch.nn as nn
from pytorch_msssim import  MS_SSIM
import cv2

class G_MS_SSIM(nn.Module):
    def __init__(self, reduction='mean'):
        super(G_MS_SSIM, self).__init__()
        self.reduction = reduction
        # Create MSSSIM module
        self.msssim = MS_SSIM(data_range=1, size_average=True, channel=1)

    def forward(self, output, target):
        # Extract green channel
        output_green = output[:, 1:2, :, :]
        target_green = target[:, 1:2, :, :]
        
        # Calculate MSSSIM
        msssim_loss = 1 - self.msssim(output_green, target_green)
        
        if self.reduction == 'mean':
            return msssim_loss.mean()
        elif self.reduction == 'sum':
            return msssim_loss.sum()
        else:
            return msssim_loss


class V_Loss(nn.Module):
    def __init__(self):
        super(V_Loss, self).__init__()

    def forward(self, input, target):
        # 将输入和目标的RGB图像转换为单通道图像，其中每个像素点的值是最大的RGB通道值
        input_max_rgb = torch.max(input, dim=1, keepdim=True)[0]
        target_max_rgb = torch.max(target, dim=1, keepdim=True)[0]

        # 计算转换后的图像之间的L1损失
        l1_loss = nn.L1Loss()(input_max_rgb, target_max_rgb)

        return l1_loss