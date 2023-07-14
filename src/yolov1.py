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
"""network"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp

from src.head import TOODHead
from src.fpn_neck import FPN
from src.resnet import resnet18
from src.config import DefaultConfig

import numpy as np


class YOLOv1(nn.Cell):
    def __init__(self, config=None, preckpt_path=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.backbone = resnet18(pretrained=config.pretrained, preckpt_path=preckpt_path)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.head = TOODHead(config.fpn_out_channels, config.class_num,
                                  config.use_GN_head,  config.prior)
        self.config = config

        self.freeze()

    def train(self, mode=True):
        """
        set module training mode, and frozen bn
        """
        super().train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters():
                    p.requires_grad = False
        if self.config.freeze_bn:
            self.apply(freeze_bn)
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)

    def flatten(self, nested):
        try:
            try:
                nested + ''
            except TypeError:
                pass
            else:
                raise TypeError

            for sublist in nested:
                for element in self.flatten(sublist):
                    yield element
        except TypeError:
            yield nested

    def freeze(self):

        for i in self.trainable_params():
            if i.name.find('bn') != -1 or i.name.find('down_sample_layer.1') != -1:
                i.requires_grad = False

        self.backbone.freeze_stages(1)


    # @ms.jit   # 静态图封装
    def construct(self, x):
        """
        Returns
        list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        """
        C3_4_5 = self.backbone(x)
        all_P = self.fpn(C3_4_5)
        cls_scores, reg_preds, anchors = self.head((all_P))
        return (cls_scores, reg_preds, anchors)

'''推理时使用该头'''
class Decode(nn.Cell):  ## inference decode
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def construct(self, inputs):
        '''
        inputs  list [cls_preds ,reg_preds]
        cls_preds  list contains five [batch_size,class_num,h,w],
        reg_preds   list contains five [batch_size,4,h,w]
        anchor
        '''

        cast = ops.Cast()
        cls_preds = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]  cls_preds[0,15200+2317].max()
        # cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds = self._reshape_cat_out([i*j[0] for i, j in zip(inputs[1], self.strides)], self.strides)  # [batch_size,sum(_h*_w),4]\  reg_preds[0,15200+2317]

        cls_classes, cls_scores = ops.ArgMaxWithValue(axis=-1)(cls_preds)  # [batch_size,sum(_h*_w)]

        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)] 验证集的类别标签从1开始

        # boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]
        boxes = reg_preds
        if self.max_detection_boxes_num > cls_scores.shape[-1]:
            max_num = cls_scores.shape[-1]
        else:
            max_num = self.max_detection_boxes_num
        topk = ops.TopK(sorted=True)
        topk_ind = topk(cls_scores, max_num)[1]   # [batch_size,max_num]

        _cls_scores = ()
        _cls_classes = ()
        _boxes = ()
        stack = ms.ops.Stack(axis=0)
        for batch in range(cls_scores.shape[0]):
            topk_index = cast(topk_ind, ms.int32)
            _cls_scores = _cls_scores + (cls_scores[batch][topk_index],)  # [max_num]
            _cls_classes = _cls_classes + (cls_classes[batch][topk_index],)  # [max_num]
            _boxes = _boxes + (boxes[batch][topk_index],)  # [max_num,4]
        cls_scores_topk = stack(_cls_scores)  # [batch_size,max_num]
        cls_classes_topk = stack(_cls_classes)  # [batch_size,max_num]
        boxes_topk = stack(_boxes)  # [batch_size,max_num,4]
        return cls_scores_topk, cls_classes_topk, boxes_topk


    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        concat = ops.Concat(axis=-1)
        boxes = concat((x1y1, x2y2))  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = ()
        reshape = ops.Reshape()
        transpose = ops.Transpose()
        for pred, stride in zip(inputs, strides):
            input_perm = (0, 2, 3, 1)
            pred = transpose(pred, input_perm)
            pred = reshape(pred, (batch_size, -1, c))
            out = out + (pred,)
        return ops.Concat(axis=1)(out)

class YOLOv1Detector(nn.Cell):
    def __init__(self, mode, config=None, preckpt_path=None):
        super().__init__()
        config = DefaultConfig
        self.mode = mode
        self.yolov1 = YOLOv1(config=config, preckpt_path=preckpt_path)
        if mode == "training":
            pass
        elif mode == "inference":
            self.decode = Decode(config.score_threshold, config.nms_iou_threshold, \
            config.max_detection_boxes_num, config.strides, config)

    # @ms.jit
    def construct(self, input_imgs):
        '''
        inputs
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''
        out = self.yolov1(input_imgs)
        # (cls_logits, reg_preds, anchors)
        if self.mode != "training":
            scores, classes, boxes = self.decode(out)
            return (scores, classes, boxes)
        return out
