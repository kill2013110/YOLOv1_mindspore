# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np
from mindspore.ops import deformable_conv2d
import mindspore.ops as ops
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

offsets = np.concatenate((offsets_x, offsets_y, mask), axis=1)
offsets = offsets.reshape((batch, 3 * deformable_groups * kh * kw, out_h, out_w))

x = Tensor(x)
weight = Tensor(weight)
offsets = Tensor(offsets)
output = ops.deformable_conv2d(x, weight, offsets, (kh, kw), (1, 1, stride_h, stride_w,), (pad_h, pad_h, pad_w, pad_w), dilations=(1, 1, dilation_h, dilation_w), deformable_groups=deformable_groups)
print(output)
# [[[[-0.00220442  0.        ]]]]