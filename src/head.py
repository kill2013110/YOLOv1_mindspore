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
"""Head"""
import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
import numpy as np

from src.config import DefaultConfig

from collections import OrderedDict

class ScaleExp(nn.Cell):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = ms.Parameter(ms.Tensor([init_value], dtype=ms.float32))

    def construct(self, x):
        return ops.Exp()(x * self.scale)

class Scale(nn.Cell):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = ms.Parameter(ms.Tensor(init_value, dtype=ms.float32), name='Scale')

    def construct(self, x):
        return x * self.scale

def bias_init_with_prob(shape, prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value.
        use in the cls branch
    """
    bias_init = np.float32(-np.log((1 - prior_prob) / prior_prob))
    return Tensor(np.array(np.ones(shape).astype(np.float32)*bias_init))

def bias_init_zeros(shape):
    """Bias zeros init method."""
    return Tensor(np.array(np.zeros(shape).astype(np.float32)))

def _conv2d_pad(in_channels, out_channels, kernel_size, stride, padding=0, pad_mode='pad',
            has_bias=True, bias_init_method=bias_init_zeros):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = initializer("normal", shape=shape, dtype=mstype.float32).init_data()  # default std=0.01
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
            has_bias=True, bias_init_method=bias_init_zeros):
    """Conv2D wrapper.
       手动pad， 对齐pytorch的zeros pad
    """
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = initializer("normal", shape=shape, dtype=mstype.float32).init_data()  # default std=0.01
    if has_bias:
        shape_bias = (out_channels,)
        biass = bias_init_method(shape_bias)
        if padding == 0:
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                             pad_mode='pad', weight_init=weights, has_bias=has_bias, bias_init=biass)
        if padding > 0:
            return nn.SequentialCell(
                nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding))),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                          pad_mode='pad', weight_init=weights, has_bias=has_bias, bias_init=biass)
            )
    else:
        if padding == 0:
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                             pad_mode='pad', weight_init=weights, has_bias=has_bias)
        if padding > 0:
            return nn.SequentialCell(
                nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding))),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                          pad_mode='pad', weight_init=weights, has_bias=has_bias)
            )

if DefaultConfig.manual_pad:
    _conv2d = _conv2d_zeros_pad
else:
    _conv2d = _conv2d_pad

class TaskDecomposition(nn.Cell):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 conv_cfg=None,
                 norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.SequentialCell([
            _conv2d(self.in_channels, self.in_channels // la_down_rate, 1, stride=1,
                      pad_mode='pad', padding=0, has_bias=True),
            nn.ReLU(),
            _conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1, stride=1,
                pad_mode='pad', padding=0, has_bias=True),
            nn.Sigmoid()
        ])

        self.reduction_conv = nn.CellList([
            _conv2d(
                self.in_channels, self.feat_channels, 1, stride=1,
                padding=0, pad_mode='pad',
                has_bias=False),
            nn.GroupNorm(32, self.feat_channels),
            nn.ReLU()
        ])

    def construct(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = ms.ops.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv[0].weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = ms.ops.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv[1](feat)  # norm层
        feat = self.reduction_conv[2](feat)  # act层

        return feat

class TOODHead(nn.Cell):
    """TOODHead used in `TOOD: Task-aligned One-stage Object Detection.

    <https://arxiv.org/abs/2108.07755>`_.

    TOOD uses Task-aligned head (T-head) and is optimized by Task Alignment
    Learning (TAL).

    Args:
        num_dcn (int): Number of deformable convolution in the head.
            Default: 0.
        anchor_type (str): If set to `anchor_free`, the head will use centers
            to regress bboxes. If set to `anchor_based`, the head will
            regress bboxes based on anchors. Default: `anchor_free`.

    Example:
        >>> self = TOODHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """
    def __init__(self,
                in_channels,
                num_classes,
                use_GN_head=True,
                prior=0.01,
                feat_channels=256,
                num_dcn=0,
                anchor_type='anchor_free',
                stacked_convs=6,
                conv_cfg=None,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 strides=DefaultConfig.strides,

                 ):
        super(TOODHead, self).__init__()
        # assert anchor_type in ['anchor_free', 'anchor_based']
        assert anchor_type in ['anchor_free', ]
        self.num_base_priors = 1
        self.cls_out_channels = num_classes

        self.in_channels = in_channels
        self.num_classes = num_classes


        self.feat_channels = feat_channels
        self.num_dcn = num_dcn
        self.anchor_type = anchor_type
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.strides = strides
        # self.single_level_grid_priors
        """Initialize layers of the head."""
        self.relu = nn.ReLU()
        self.inter_convs = nn.CellList()
        # self.inter_convs = []

        for i in range(self.stacked_convs):
            assert num_dcn == 0  #暂不支持dcn
            chn = self.in_channels if i == 0 else self.feat_channels

            ConvModule = nn.SequentialCell([
                _conv2d(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    pad_mode='pad',
                    padding=1,
                    has_bias=False,),
                nn.GroupNorm(self.norm_cfg['num_groups'], self.feat_channels),
                nn.ReLU()
            ])

            self.inter_convs.append(ConvModule)

        self.cls_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)
        self.reg_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)

        self.tood_cls = _conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels, 3, stride=1,
            pad_mode='pad', padding=1,
            has_bias=True, bias_init_method=bias_init_with_prob,
        )
        self.tood_reg = _conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, stride=1,
            pad_mode='pad', padding=1,
            has_bias=True,
        )

        self.cls_prob_module = nn.SequentialCell([
            _conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1, stride=1,
                      pad_mode='pad', padding=0,
                      has_bias=True),
            nn.ReLU(),
            _conv2d(self.feat_channels // 4, 1, 3, stride=1,
                      pad_mode='pad', padding=1,
                      has_bias=True, bias_init_method=bias_init_with_prob)
        ])
        self.reg_offset_module = nn.SequentialCell([
            _conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1, stride=1,
                      pad_mode='pad', padding=0,
                      has_bias=True),
            nn.ReLU(),
            _conv2d(self.feat_channels // 4, 4 * 2, 3, stride=1,
                      pad_mode='pad', padding=1,
                      has_bias=True)
        ])

        self.scales = nn.CellList(
            # [ScaleExp(1.0) for _ in self.prior_generator.strides])
            [Scale(1.0) for _ in range(5)])


    def construct(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Decoded box for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_anchors * 4. In [tl_x, tl_y, br_x, br_y] format.
        """
        # import numpy as np
        # import torch, copy
        # a = torch.load('/mnt/f/liwenlong/mask_mmdet/feats.pth', map_location=torch.device('cpu'))['data']
        # a = tuple([ms.Tensor(i) for i in a])
        # feats = copy.copy(a)



        cls_scores = []
        bbox_preds = []
        anchors = []
        for idx in range(5):
            x = feats[idx]
            scale = self.scales[idx]
            stride = self.strides[idx]
            b, c, h, w = x.shape

            anchor = self.stride2anchor(h, w, stride)
            anchors.append(anchor) ##返回值不需要重复的anchor

            anchor = ms.ops.cat([anchor for _ in range(b)])
            # extract task interactive features
            inter_feats = []
            for j in range(len(self.inter_convs)):  # 6
                x = self.inter_convs[j](x)
                inter_feats.append(x)
            feat = ms.ops.cat(inter_feats, 1)

            # task decomposition
            avg_feat = ms.ops.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)

            # cls prediction and alignment
            cls_logits = self.tood_cls(cls_feat)
            cls_prob = self.cls_prob_module(feat)
            cls_score = ms.ops.sqrt(ms.ops.sigmoid(cls_logits) * ms.ops.sigmoid(cls_prob))

            # reg prediction and alignment
            # if self.anchor_type == 'anchor_free':
            reg_dist = scale(self.tood_reg(reg_feat).exp()).float()
            reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
            reg_bbox = distance2bbox(
                self.anchor_center(anchor) / stride[0], reg_dist).reshape(b, h, w, 4).permute(0, 3, 1, 2)  # (b, c, h, w)

            reg_offset = self.reg_offset_module(feat)
            bbox_pred = self.deform_sampling(reg_bbox, reg_offset)
            # bbox_pred = self.deform_sampling_torch(reg_bbox,reg_offset)

            # After deform_sampling, some boxes will become invalid (The
            # left-top point is at the right or bottom of the right-bottom
            # point), which will make the GIoULoss negative.
            invalid_bbox_idx = (bbox_pred[:, [0]] > bbox_pred[:, [2]]) | \
                               (bbox_pred[:, [1]] > bbox_pred[:, [3]])
            invalid_bbox_idx = invalid_bbox_idx.expand_as(bbox_pred)
            bbox_pred = ms.ops.where(invalid_bbox_idx, reg_bbox, bbox_pred)

            cls_scores.append(cls_score)  # cls_scores[1][0,:,30,37].max()
            bbox_preds.append(bbox_pred)  # bbox_preds[1][0,:,30,37]
        return tuple(cls_scores), tuple(bbox_preds), tuple(anchors)
    def deform_sampling_torch(self, feat, offset):
        import numpy as np
        import torch, copy, torchvision
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for feature sampling
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w = feat.shape
        weight = feat.new_ones((c, 1, 1, 1), dtype=ms.float32)
        y = torchvision.ops.deform_conv2d(torch.tensor(feat.asnumpy()), torch.tensor(offset.asnumpy()), torch.tensor(weight.asnumpy()), stride=1, padding=0, dilation=1)
        return ms.Tensor(y.numpy())

    def deform_sampling(self, feat, offsets):
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for feature sampling
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w = feat.shape
        weight = feat.new_ones((c, 1, 1, 1), dtype=ms.float32)
        '''mmcv.DeformConv2dFunction == torchvision.ops.deform_conv2d
            torchvision.ops.deform_conv2d与mindspore.deformable_conv2d 的区别见
            https://www.mindspore.cn/docs/zh-CN/r2.0/note/api_mapping/pytorch_diff/deform_conv2d.html
            其中的代码示例
        '''
        x_y = offsets.reshape((b, c, 1, 1, 2, h, w)).transpose(0, 4, 1, 2, 3, 5, 6)
        mask = ms.ops.ones([b, 1, c, 1, 1, h, w], ms.float32)
        y_x_mask = ms.ops.cat((x_y[:, ::-1], mask), axis=1)
        y_x_mask = y_x_mask.reshape((b, 3 * c * 1 * 1, h, w))
        y = ms.ops.deformable_conv2d(x=feat, weight=weight, offsets=y_x_mask, kernel_size=(1, 1),
                                     strides=(1, 1, 1, 1), padding=(0, 0, 0, 0), dilations=(1, 1, 1, 1), groups=c, deformable_groups=c)
        return y

    '''
    x_y = offsets.reshape((1, c, 1, 1, 2, h, w)).transpose(0, 4, 1, 2, 3, 5, 6)
    mask = ms.ops.ones([1, 1, c, 1, 1, h, w], ms.float32)
    y_x_mask = ms.ops.cat((x_y[:, 1],x_y[:, 0], mask), axis=1)
    y_x_mask = y_x_mask.reshape((1, 3 * c * 1 * 1, h, w))
    y_x_mask.shape
    '''
    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return ms.ops.stack([anchors_cx, anchors_cy], axis=-1)

    def stride2anchor(self, h, w, stride):
        # anchor = []
        stride_w, stride_h = stride[0]*4, stride[1]*4
        # for i in range(h):
        #     for j in range(w):
        #         anchor.append([j*stride[0]-stride_w, i*stride[1]-stride_h,
        #                        j*stride[0]+stride_w, i*stride[1]+stride_h])
        # anchor = ms.Tensor(anchor)
        base = ms.ops.meshgrid(ms.ops.range(ms.Tensor(0), ms.Tensor(w*stride[0]), ms.Tensor(stride[0])),
                               ms.ops.range(ms.Tensor(0), ms.Tensor(h*stride[0]), ms.Tensor(stride[0]))
                               )
        anchor = ms.ops.cat((base[0].reshape(-1, 1)-stride_w,
                             base[1].reshape(-1, 1)-stride_h,
                             base[0].reshape(-1, 1)+stride_w,
                             base[1].reshape(-1, 1)+stride_h),
                            axis=1)
        return anchor


# def base_anchor():
#     pass

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = ms.ops.stack([x1, y1, x2, y2], -1)


    # if max_shape is not None:
    #     if bboxes.dim() == 2 and not ms.onnx.is_in_onnx_export():
    #         # speed up
    #         bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
    #         bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
    #         return bboxes
    #
    #     # clip bboxes with dynamic `min` and `max` for onnx
    #     if torch.onnx.is_in_onnx_export():
    #         from mmdet.core.export import dynamic_clip_for_onnx
    #         x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
    #         bboxes = torch.stack([x1, y1, x2, y2], axis=-1)
    #         return bboxes
    #     if not isinstance(max_shape, torch.Tensor):
    #         max_shape = x1.new_tensor(max_shape)
    #     max_shape = max_shape[..., :2].type_as(x1)
    #     if max_shape.ndim == 2:
    #         assert bboxes.ndim == 3
    #         assert max_shape.size(0) == bboxes.size(0)
    #
    #     min_xy = x1.new_tensor(0)
    #     max_xy = torch.cat([max_shape, max_shape],
    #                        dim=-1).flip(-1).unsqueeze(-2)
    #     bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
    #     bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes