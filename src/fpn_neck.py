# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""FPN"""
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import ResizeNearestNeighbor
from mindspore.common.initializer import initializer, HeUniform, XavierUniform
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from src.config import DefaultConfig

def bias_init_kaiming(shape):
    return initializer(HeUniform(), shape=(shape[0], 1), dtype=mstype.float32).init_data()[:, 0]

def bias_init_xavier(shape):
    """Bias init method."""
    return Tensor(np.array(np.zeros(shape).astype(np.float32)))

def _conv2d_pad(in_channels, out_channels, kernel_size, stride, padding=0, pad_mode='pad',
            has_bias=True, bias_init_method=bias_init_kaiming):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = initializer(HeUniform(negative_slope=1), shape=shape, dtype=mstype.float32).init_data()  # tood default

    if has_bias:
        shape_bias = (out_channels,)
        biass = bias_init_method(shape_bias)
        return nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         pad_mode=pad_mode, weight_init=weights, has_bias=has_bias,
                         bias_init=biass)
    else:
        return nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         pad_mode=pad_mode, weight_init=weights, has_bias=has_bias,
                         )

def _conv2d_zeros_pad(in_channels, out_channels, kernel_size, stride, padding=0, pad_mode='pad',
            has_bias=True, bias_init_method=bias_init_kaiming):
    """Conv2D wrapper.
       对齐pytorch的zeros pad
    """
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = initializer("normal", shape=shape, dtype=mstype.float32).init_data()  # default std=0.01
    if has_bias:
        shape_bias = (out_channels,)
        biass = bias_init_method(shape_bias)
        if padding == 0:
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                             pad_mode=pad_mode, weight_init=weights, has_bias=has_bias, bias_init=biass)
        if padding > 0:
            return nn.SequentialCell(
                nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding))),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                          pad_mode=pad_mode, weight_init=weights, has_bias=has_bias, bias_init=biass)
            )
    else:
        if padding == 0:
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                             pad_mode=pad_mode, weight_init=weights, has_bias=has_bias)
        if padding > 0:
            return nn.SequentialCell(
                nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding))),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                          pad_mode=pad_mode, weight_init=weights, has_bias=has_bias)
            )
if DefaultConfig.manual_pad:
    _conv2d = _conv2d_zeros_pad
else:
    _conv2d = _conv2d_pad


class FPN(nn.Cell):
    '''only for resnet50,101,152'''

    def __init__(self, features=256, use_p5=True, in_channels=[512, 1024, 2048]):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        # self.lateral_convs = nn.CellList(
        #     _conv2d(512, features, kernel_size=1, stride=1, pad_mode='pad', padding=0),
        #     _conv2d(1024, features, kernel_size=1, stride=1, pad_mode='pad', padding=0),
        #     _conv2d(2048, features, kernel_size=1, stride=1, pad_mode='pad', padding=0),
        # )
        #
        # self.fpn_convs = nn.CellList(
        #     _conv2d(features, features, kernel_size=3, stride=1, pad_mode='pad', padding=1),
        #     _conv2d(features, features, kernel_size=3, stride=1, pad_mode='pad', padding=1),
        #     _conv2d(features, features, kernel_size=3, stride=1, pad_mode='pad', padding=1),
        # )
        self.lateral_convs = nn.CellList([
            _conv2d(512, features, kernel_size=1, stride=1, pad_mode='pad', padding=0),
            _conv2d(1024, features, kernel_size=1, stride=1, pad_mode='pad', padding=0),
            _conv2d(2048, features, kernel_size=1, stride=1, pad_mode='pad', padding=0),]
        )

        self.fpn_convs = nn.CellList([
            _conv2d(features, features, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            _conv2d(features, features, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            _conv2d(features, features, kernel_size=3, stride=1, pad_mode='pad', padding=1),]
        )

        if use_p5:
            self.fpn_convs.append(
                _conv2d(features, features, kernel_size=3, pad_mode='pad', padding=1, stride=2))
        else:
            self.fpn_convs.append(
                _conv2d(2048, features, kernel_size=3, pad_mode='pad', padding=1, stride=2))

        self.fpn_convs.append(
            _conv2d(features, features, kernel_size=3, pad_mode='pad', padding=1, stride=2))
        self.use_p5 = use_p5
        # constant_init = mindspore.common.initializer.Constant(0)
        # for i in self.lateral_convs:
        #     constant_init(i.bias)
        # for i in self.fpn_convs:
        #     constant_init(i.bias)

    def upsamplelike(self, inputs):
        src, target = inputs
        resize = ResizeNearestNeighbor((target.shape[2], target.shape[3]))
        return resize(src)

    def construct(self, inputs):

        C3, C4, C5 = inputs

        P3 = self.lateral_convs[0](C3)  # 512
        P4 = self.lateral_convs[1](C4)  # 1024
        P5 = self.lateral_convs[2](C5)  # 2048

        P4 = P4 + self.upsamplelike((P5, C4))
        P3 = P3 + self.upsamplelike((P4, C3))

        P3 = self.fpn_convs[0](P3)
        P4 = self.fpn_convs[1](P4)
        P5 = self.fpn_convs[2](P5)
        P5 = P5 if self.use_p5 else C5
        P6 = self.fpn_convs[3](P5)
        relu = ops.ReLU()
        P7 = self.fpn_convs[4](relu(P6))
        return (P3, P4, P5, P6, P7)
