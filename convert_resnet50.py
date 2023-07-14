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
"""FCOS EVAL"""
import json
import os
import re
import argparse
import cv2
import numpy as np
import mindspore as ms

from mindspore import context, nn

from src.tood import TOODDetector

import copy


from src.fpn_neck import FPN
from src.resnet import resnet50


import torch


parser = argparse.ArgumentParser()
parser.add_argument("--device_id", type=int, default=0, help="DEVICE_ID to run ")
parser.add_argument("--eval_path", type=str,
                    default='/mnt/f/datasets/coco2017/images/val2017',
                    )
parser.add_argument("--anno_path", type=str,
                    default='/mnt/f/datasets/coco2017/annotations/instances_val2017.json',
                    )
parser.add_argument("--ckpt_path", type=str,
                    default="/mnt/f/pretrain_weight/tood_r50_fpn_1x_coco.pth",
                    )

opt = parser.parse_args()

def resnet50_pth2ckpt(ms_model, pth_path):
    # ms_ckpt = ms.load_checkpoint('tood_ms_zerospad.ckpt')  # mindspore FCOS保存的随机权重
    ms_ckpt = ms_model  # mindspore FCOS保存的随机权重
    pth = torch.load(pth_path, map_location=torch.device('cpu'))  # pytorch FCOS权重
    match_pt_kv = {}  # 匹配到的pt权重的name及value的字典
    match_pt_kv_mslist = []  # 匹配到的pt权重的name及value的字典, mindspore加载权重需求的格式
    not_match_pt_kv = {}  # 未匹配到的pt权重的name及value
    matched_ms_k = []  # 被匹配到的ms权重名称

    '''一般性的转换规则'''
    pt2ms = {'backbone': 'tood_body.backbone',  # backbone部分
             'neck': 'tood_body.fpn',
             'bbox_head': 'tood_body.head',
             'downsample': 'down_sample_layer',
             }

    '''conv层的转换规则, 一致，可忽略'''
    pt2ms_conv = {
        "weight": "weight",
        "bias": "bias",
    }

    '''downsample层的转换规则, 有卷积层和bn层, 分别为0，1命名，在torch中weight重复'''
    pt2ms_down = {
        "0.weight": "0.weight",
        "1.weight": "1.gamma",

        "1.bias": "1.beta",
        "running_mean": "moving_mean",
        "running_var": "moving_variance",
    }

    '''BN层的转换规则'''
    pt2ms_bn = {
        "running_mean": "moving_mean",
        "running_var": "moving_variance",
        "weight": "gamma",
        "bias": "beta",
    }

    '''GN层的转换规则'''
    pt2ms_gn = {
        "weight": "gamma",
        "bias": "beta",
    }

    for i, v in pth.items():
        pt_name = copy.deepcopy(i)
        pt_value = copy.deepcopy(v)
        '''一般性的处理'''
        for k, v in pt2ms.items():
            if k in pt_name:
                pt_name = pt_name.replace(k, v)

        '''conv层的转换规则, 一致，可忽略'''

        '''FPN部分特别处理'''
        if 'fpn' in pt_name:
            pt_name = pt_name.replace('.conv', '')

        '''下采样层特别处理'''
        if 'down' in pt_name:
            for k, v in pt2ms_down.items():
                if k in pt_name:
                    pt_name = pt_name.replace(k, v)

        '''BN层处理'''
        if 'bn' in pt_name:
            for k, v in pt2ms_bn.items():
                if k in pt_name:
                    pt_name = pt_name.replace(k, v)

        '''GN层处理'''
        if 'gn' in pt_name:
            for k, v in pt2ms_gn.items():
                if k in pt_name:
                    pt_name = pt_name.replace(k, v)

        '''reduction_conv 和inter_convs 因为静态图重构了模型，需要特别处理'''
        if 'reduction_conv' in pt_name or 'inter_convs' in pt_name:
            if '.conv.' in pt_name:
                pt_name = pt_name.replace('.conv.', '.0.')
            elif '.gn.' in pt_name:
                pt_name = pt_name.replace('.gn.', '.1.')

        '''改名成功，匹配到ms中的权重了，记录'''
        if pt_name in ms_ckpt.keys():
            assert pt_name not in matched_ms_k  # 不能重复匹配
            if not 'scale' in pt_name:
                assert pt_value.shape == ms_ckpt[pt_name].shape
            match_pt_kv[pt_name] = pt_value
            match_pt_kv_mslist.append({'name': pt_name, 'data': ms.Tensor(pt_value.detach().numpy(), ms_ckpt[pt_name].dtype)})
            matched_ms_k.append(pt_name)

        # 由于手写 zeros pad对齐，导致mindspore有pad的卷积层命名发生改变,在没有手动pad的时候无需以下两个elif
        elif '.weight' in pt_name:
            if pt_name.replace('.weight', '.1.weight') in ms_ckpt.keys():
                pt_name = pt_name.replace('.weight', '.1.weight')
                assert pt_value.shape == ms_ckpt[pt_name].shape
                match_pt_kv[pt_name] = pt_value
                match_pt_kv_mslist.append(
                    {'name': pt_name, 'data': ms.Tensor(pt_value.detach().numpy(), ms_ckpt[pt_name].dtype)})
                matched_ms_k.append(pt_name)
        elif '.bias' in pt_name:
            if pt_name.replace('.bias', '.1.bias') in ms_ckpt.keys():
                pt_name = pt_name.replace('.bias', '.1.bias')
                assert pt_value.shape == ms_ckpt[pt_name].shape
                match_pt_kv[pt_name] = pt_value
                match_pt_kv_mslist.append(
                    {'name': pt_name, 'data': ms.Tensor(pt_value.detach().numpy(), ms_ckpt[pt_name].dtype)})
                matched_ms_k.append(pt_name)
        else:
            not_match_pt_kv[i + '   ' + pt_name] = pt_value

    '''打印未匹配的pt权重名称'''
    print('\n\n-----------------------------未匹配的pt权重名称----------------------------')
    print('----------原名称--------                        ----------转换后名称---------')
    for j, v in not_match_pt_kv.items():
        # if not 'num_batches_tracked' in j:
            print(j, np.array(v.shape))

    '''打印未被匹配到的ms权重名称'''
    print('\n\n---------------------------未被匹配到的ms权重名称----------------------------')
    for j, v in ms_ckpt.items():
        if j not in matched_ms_k:
            print(j, np.array(v.shape))
    print('end')
    return match_pt_kv_mslist

if __name__ == "__main__":

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=opt.device_id)

    resnet50 = resnet50(pretrained=False)

    match_pt_kv_mslist = resnet50_pth2ckpt(resnet50.parameters_dict(), pth_path="/mnt/f/pretrain_weight/resnet50-0676ba61.pth")

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=opt.device_id)
    ms.save_checkpoint(match_pt_kv_mslist, 'resnet50_pth2ms_jit.ckpt')
    ms.load_checkpoint('temp.ckpt', resnet50)
    input = ms.ops.ones((1, 3, 640, 640), ms.float32)
    out = resnet50(input)





