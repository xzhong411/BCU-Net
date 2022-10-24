""" Parts of the U-Net model """
from torchvision.transforms import ToPILImage, transforms

"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv(nn.Module):
    """(convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_Cat(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
        )
        self.doubleconv = DoubleConv(in_channels+out_channels, out_channels)
    def forward(self, x1,x2):
        x1=self.maxpool_conv(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.doubleconv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ca=ChannelAttention(out_channels)
        self.sa=SpatialAttention()
        self.conv=Conv(in_channels,out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # DoubleConv(in_channels, out_channels)
        )
    def forward(self, x1):
        x = self.maxpool_conv(x1)
        x0 = self.conv(x)
        x = self.ca(x0) * x0
        x = self.sa(x) * x

        # if self.downsample is not None:
        #     residual = self.downsample(x1)

        x += x0
        return self.relu(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
        # self.contra=nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        # 这个上采用需要设置其输入通道，输出通道.其中kernel_size、stride
        # 大小要跟对应下采样设置的值一样大小。这样才可恢复到相同的wh。这里时反卷积操作。
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        # 这里不会改变通道数，其中scale_factor是上采用的放大因子，其是相对于当前的输入大小的倍数
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=(in_channels//32), align_corners=True))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Sequential(nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1),
        nn.BatchNorm2d(out_channels))
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Sequential(
            upconv2x2(in_channels, out_channels,mode=""),
            conv1x1(in_channels, out_channels),
        )
        # nn.Upsample(scale_factor=in_channels//32, mode='nearest'),
        # nn.Conv2d(out_channels,out_channels, kernel_size=1)

    def forward(self, x):
        x=self.up(x)
        outlayer = nn.Sigmoid()
        return outlayer(x)


class OutConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels, kernel_size=1),
        )

    def forward(self, x):
        x=self.conv(x)
        outlayer = nn.Sigmoid()
        return outlayer(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)




class Weighted_Cross_Entropy_Loss(nn.Module):
    def __init__(self):
        super(Weighted_Cross_Entropy_Loss, self).__init__()

    def forward(self, inputs, target, num_classes=2, weighted=False, softmax=False):
        """
        input  : NxCxHxW Variable
        target :  NxHxW LongTensor
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).contiguous()
        if weighted:
            ce_weight = 1 - (torch.sum(target_onehot, dim=(0, 2, 3)).float() / torch.sum(target_onehot).float())
            ce_weight = ce_weight.view(1, num_classes, 1, 1)
        else:
            ce_weight = 1
        loss = - 1.0 * torch.sum(ce_weight * target_onehot * torch.log(inputs.clamp(min=0.005, max=1)), dim=1)
        loss = loss.mean()
        return loss


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        # input（4，2，512，512）
        # target（4，2，512，512） --> (4,512,512)
        target = target.long()
        b,_,w,h = target.shape
        new_target = torch.ones(size=(b,w,h)).to(input.device)
        for i in range(b):
            new_target[i][target[0][0]==1] = 0
            new_target[i][target[0][1]==1] = 1# 2
        logp = self.ce(input, new_target.long())
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


def diceCoeff(pred, gt, smooth=1, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = 2 * (intersection + smooth) / (unionset + smooth)

    return loss.sum() / N
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, target, num_classes=2, return_hard_dice=False, softmax=False):
        """
        input  : NxCxHxW Variable
        target :  NxHxW LongTensor

        """
        assert inputs.dim() == 4, "Input must be a 4D Tensor."
        # target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).contiguous()
        # target_onehot = F.one_hot(target.long(), num_classes=num_classes).contiguous()
        target_onehot=target

        assert inputs.size() == target_onehot.size(), "Input sizes must be equal."

        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        # binary dice
        if inputs.shape[1] == 2:
            intersection = torch.sum(inputs * target_onehot, [2, 3])
            L = torch.sum(inputs, [2, 3])
            R = torch.sum(target_onehot, [2, 3])
            coefficient = (2 * intersection + 1e-6) / (L + R + 1e-6)
            loss = 1 - coefficient.mean()

        else:
            intersection = torch.sum(inputs * target_onehot, (0, 2, 3))
            L = torch.sum(inputs * inputs, (0, 2, 3))
            R = torch.sum(target_onehot * target_onehot, (0, 2, 3))

            coefficient = (2 * intersection + 1e-6) / (L + R + 1e-6)
            loss = 1 - coefficient.mean()

        if return_hard_dice:
            pred_hard = torch.argmax(inputs, dim=1)
            pred_onehot = F.one_hot(pred_hard, num_classes=num_classes).permute(0, 3, 1, 2).contiguous()
            hard_intersection = torch.sum(pred_onehot * target_onehot, (0, 2, 3))
            hard_cardinality = torch.sum(pred_onehot + target_onehot, (0, 2, 3))
            hard_dice_loss = 1 - (2 * hard_intersection + 1e-6) / (hard_cardinality + 1e-6)
            hard_dice_loss = hard_dice_loss.mean()
            return loss, hard_dice_loss
        else:
            return loss