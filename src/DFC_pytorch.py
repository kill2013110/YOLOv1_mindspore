# PyTorch
import torch
from torch import tensor
import numpy as np
from torchvision.ops import deform_conv2d
np.random.seed(1)
kh, kw = 1, 1
batch = 1
deformable_groups = 4
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1
pad_h, pad_w = 0, 0
x_h, x_w = 1, 2
out_h = (x_h + 2 * pad_h - dilation_h * (kh - 1) - 1) // stride_h + 1
out_w = (x_w + 2 * pad_w - dilation_w * (kw - 1) - 1) // stride_w + 1

x = np.random.randn(batch, 64, x_h, x_w).astype(np.float32)
weight = np.random.randn(batch, 64, kh, kw).astype(np.float32)
offsets_x = np.random.randn(batch, 1, deformable_groups, kh, kw, out_h, out_w).astype(np.float32)
offsets_y = np.random.randn(batch, 1, deformable_groups, kh, kw, out_h, out_w).astype(np.float32)
mask = np.random.randn(batch, 1, deformable_groups, kh, kw, out_h, out_w).astype(np.float32)

offsets = np.concatenate((offsets_y, offsets_x), axis=1)
offsets = offsets.transpose(0, 2, 3, 4, 1, 5, 6)
offsets = offsets.reshape((batch, 2 * deformable_groups * kh * kw, out_h, out_w))
mask = mask.transpose(0, 2, 3, 4, 1, 5, 6)
mask = mask.reshape((batch, 1 * deformable_groups * kh * kw, out_h, out_w))
x = torch.from_numpy(x.copy().astype(np.float32))
weight = torch.from_numpy(weight.copy().astype(np.float32))
offsets = torch.from_numpy(offsets.copy().astype(np.float32))
mask = torch.from_numpy(mask.copy().astype(np.float32))
output = deform_conv2d(x, offsets, weight, stride=(stride_h, stride_w), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), mask=mask)
print(output)
# tensor([[[[-0.0022,  0.0000]]]])