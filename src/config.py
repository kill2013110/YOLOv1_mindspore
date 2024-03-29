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
"""Config"""
CUR_EPOCH = 0

class DefaultConfig():
    manual_pad = False
    if manual_pad:
        print('------------------------ model construct use manual_pad !--------------------------')

    #backbone
    pretrained = True
    freeze_stage_1 = True
    freeze_bn = False
    #fpn
    fpn_out_channels = 256
    use_p5 = True
    #head
    class_num = 80
    use_GN_head = True
    prior = 0.01
    #training
    initial_epoch = 4
    strides = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
    save_checkpoint = True
    pos_weight = -1
    allowed_border = -1
    #inference
    score_threshold = 0.05
    nms_iou_threshold = 0.6
    max_detection_boxes_num = 1000
