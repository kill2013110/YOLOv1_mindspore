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
"""network_define"""

import mindspore.nn as nn

from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
import mindspore as ms

from mindspore import Tensor

from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean, _get_parallel_mode)
from mindspore.context import ParallelMode

from src.config import DefaultConfig
from src.assigner_utils import sample, images_to_levels
from src.assigner import ATSSAssigner, TaskAlignedAssigner
from src.loss import FocalLoss, GIoULoss, QualityFocalLoss

import src.config

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.
    """
    def __init__(self, model, ):
        super(WithLossCell, self).__init__(auto_prefix=False)
        # config = DefaultConfig
        self.config = DefaultConfig
        self._model = model
        self.atss_assigner = ATSSAssigner()
        self.taskaligned_assigner = TaskAlignedAssigner()

        self.initial_loss_cls = FocalLoss(activated=True)  # activated默认值为False， 本项目需要设定True
        self.loss_cls = QualityFocalLoss(activated=True)

        self.loss_bbox = GIoULoss(loss_weight=2.0)

    def construct(self, input_imgs, gt_bboxes, gt_classes, cur_epoch=0):
        self.epoch = src.config.CUR_EPOCH
        #preds
        out = self._model(input_imgs)
        (cls_scores, bbox_preds, anchor_list) = out


        '''debug'''
        # import torch
        # torch_out = torch.load('/mnt/f/liwenlong/mask_mmdet/cls_bbox_outs.pth', map_location=torch.device('cpu'))
        # cls_scores, bbox_preds = tuple([ms.Tensor(i.detach().numpy()) for i in torch_out[0]]), tuple([ms.Tensor(i.detach().numpy()) for i in torch_out[1]])


        '''batch'''
        num_imgs = len(input_imgs)

        '''对预测值的多尺度维度flatten'''
        flatten_cls_scores = ms.ops.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self._model.tood_body.head.cls_out_channels)
            for cls_score in cls_scores], 1)
        flatten_bbox_preds = ms.ops.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) * stride[0]
            for bbox_pred, stride in zip(bbox_preds, self.config.strides)], 1)

        num_level_anchors = [len(i) for i in anchor_list]  # 相比mmdet该步已经简化，没有筛掉pad部分所对应的anchor
        flat_anchors = ms.ops.cat(anchor_list, axis=0)  # 对多尺度维度flatten，同一batch anchor共用,

        '''sample assignment'''
        if self.epoch < self.config.initial_epoch:
            # 前4epoch用ATSS_sample_assigment完成样本分配和采样
            sample_assigment_result = self.ATSS_sample_assigment(self.atss_assigner.assign,
                                                            num_imgs, flat_anchors,
                                                            gt_bboxes, gt_classes,
                                                            num_level_anchors)

            (all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights) = sample_assigment_result
            all_assign_metrics = [weight[..., 0] for weight in all_bbox_weights]
        else:
            # 后面用 taskaligned_assigner
            sample_assigment_result = self.TAL_sample_assigment(self.taskaligned_assigner.assign,
                                                                num_imgs,
                                                                flatten_cls_scores,
                                                                flatten_bbox_preds,
                                                                flat_anchors,
                                                                gt_bboxes,
                                                                gt_classes,
                                                                num_level_anchors)
            (all_anchors, all_labels, all_label_weights, all_bbox_targets, all_assign_metrics) = sample_assigment_result

        if any([labels is None for labels in all_labels]):
            return None

        '''转化为多尺度维度'''
        anchor_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        norm_alignment_metrics_list = images_to_levels(all_assign_metrics, num_level_anchors)

        alignment_metrics_list = norm_alignment_metrics_list

        '''get loss'''
        # losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors
        losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = self.compute_loss(
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            alignment_metrics_list,
            self.config.strides)

        cls_avg_factor = sum(cls_avg_factors).clamp(min=1)
        losses_cls_list = [i/cls_avg_factor for i in losses_cls]

        bbox_avg_factor = sum(bbox_avg_factors).clamp(min=1)
        losses_bbox_list = [i/bbox_avg_factor for i in losses_bbox]

        loss = sum(losses_cls_list) + sum(losses_bbox_list)
        # return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        return loss

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._model

    def ATSS_sample_assigment(self, ATSSassigner, num_imgs, flat_anchors, gt_bboxes, gt_classes, num_level_anchors):
        '''输入中每个tensor的第一个维度为batch，即num_imgs，同一batch anchor共用'''

        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights = [], [], [], [], []

        for i in range(num_imgs):
            anchors = flat_anchors

            '''assign'''
            img_gt_bboxes = gt_bboxes[i][gt_classes[i] != -1]
            img_gt_classes = gt_classes[i][gt_classes[i] != -1]
            assign_result = ATSSassigner(anchors, num_level_anchors,
                                     img_gt_bboxes, gt_bboxes_ignore=None,
                                     gt_labels=img_gt_classes)
            # assign_result:(num_gt, assigned_gt_inds, max_overlaps, assigned_labels, pos_inds, neg_inds, assigned_labels)

            '''sample'''
            sample_restlt = sample(assign_result, anchors, img_gt_bboxes)

            (pos_inds, neg_inds, pos_bboxes, neg_bboxes, pos_is_gt, num_gts, pos_assigned_gt_inds, pos_gt_bboxes,
             pos_gt_labels) = sample_restlt

            num_valid_anchors = anchors.shape[0]
            bbox_targets = ms.ops.zeros_like(anchors, dtype=ms.float32)
            bbox_weights = ms.ops.zeros_like(anchors, dtype=ms.float32)

            labels = ms.ops.ones((num_valid_anchors,), dtype=ms.int64) * self.config.class_num
            label_weights = anchors.new_zeros(num_valid_anchors, dtype=ms.float32)

            if len(pos_inds) > 0:
                pos_bbox_targets = pos_gt_bboxes  # tood的box已经decode
                bbox_targets[pos_inds, :] = pos_bbox_targets
                bbox_weights[pos_inds, :] = 1.0
                if img_gt_classes is None:
                    # Only rpn gives gt_labels as None
                    # Foreground is the first class since v2.5.0
                    labels[pos_inds] = 0
                else:
                    labels[pos_inds] = img_gt_classes[pos_assigned_gt_inds]
                if self.config.pos_weight <= 0:
                    label_weights[pos_inds] = 1.0
                else:
                    label_weights[pos_inds] = self.config.pos_weight
            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0

            #  map up to original set of anchors 该步由于忽略inviald anchor所以可省略

            all_anchors.append(anchors)
            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_bbox_targets.append(bbox_targets)
            all_bbox_weights.append(bbox_weights)
            # pos_inds_list.append(pos_inds)
            # neg_inds_list.append(neg_inds)

        return (all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights)

    def TAL_sample_assigment(self, TALassigner, num_imgs, flatten_cls_scores, flatten_bbox_preds,
                                  flat_anchors, gt_bboxes, gt_classes, num_level_anchors):
        '''输入中每个tensor的第一个维度为batch，即num_imgs，同一batch anchor共用'''

        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_norm_alignment_metrics = [], [], [], [], []

        for i in range(num_imgs):
            anchors = flat_anchors

            img_cls_scores = flatten_cls_scores[i]
            img_bbox_preds = flatten_bbox_preds[i]
            img_gt_bboxes = gt_bboxes[i][gt_classes[i] != -1]
            img_gt_classes = gt_classes[i][gt_classes[i] != -1]

            '''assign'''
            assign_result = TALassigner(img_cls_scores, img_bbox_preds, anchors,
                                     img_gt_bboxes, gt_bboxes_ignore=None, gt_labels=img_gt_classes)
            (num_gt, assigned_gt_inds, max_overlaps, assigned_labels, pos_inds, neg_inds, assigned_labels, assign_metrics) = assign_result

            '''sample'''
            sample_input_assign_result = (num_gt, assigned_gt_inds, max_overlaps, assigned_labels, pos_inds, neg_inds, assigned_labels)
            sample_restlt = sample(sample_input_assign_result, anchors, img_gt_bboxes)
            (pos_inds, neg_inds, pos_bboxes, neg_bboxes, pos_is_gt, num_gts, pos_assigned_gt_inds, pos_gt_bboxes,
             pos_gt_labels) = sample_restlt

            num_valid_anchors = anchors.shape[0]
            bbox_targets = ms.ops.zeros_like(anchors, dtype=ms.float32)
            labels = ms.ops.full((num_valid_anchors,), self.config.class_num, dtype=ms.int64)
            label_weights = anchors.new_zeros(num_valid_anchors, dtype=ms.float32)
            norm_alignment_metrics = anchors.new_zeros(num_valid_anchors, dtype=ms.float32)

            if len(pos_inds) > 0:
                # point-based
                pos_bbox_targets = pos_gt_bboxes
                bbox_targets[pos_inds, :] = pos_bbox_targets

                if img_gt_classes is None:
                    # Only rpn gives gt_labels as None
                    # Foreground is the first class since v2.5.0
                    labels[pos_inds] = 0
                else:
                    labels[pos_inds] = img_gt_classes[pos_assigned_gt_inds]
                if self.config.pos_weight <= 0:
                    label_weights[pos_inds] = 1.0
                else:
                    label_weights[pos_inds] = self.config.pos_weight
            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0

            class_assigned_gt_inds = ms.ops.unique(pos_assigned_gt_inds)[0]
            for gt_inds in class_assigned_gt_inds:
                gt_class_inds = pos_inds[pos_assigned_gt_inds == gt_inds]
                pos_alignment_metrics = assign_metrics[gt_class_inds]
                pos_ious = max_overlaps[gt_class_inds]
                pos_norm_alignment_metrics = pos_alignment_metrics / (
                        pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
                norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics

            #  map up to original set of anchors 该步由于忽略inviald anchor所以可省略

            all_anchors.append(anchors)
            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_bbox_targets.append(bbox_targets)
            all_norm_alignment_metrics.append(norm_alignment_metrics)

        return (all_anchors, all_labels, all_label_weights, all_bbox_targets, all_norm_alignment_metrics)

    def compute_loss(self, anchor_list,
                            cls_scores,
                            bbox_preds,
                            labels_list,
                            label_weights_list,
                            bbox_targets_list,
                            alignment_metrics_list,
                            strides,
                            ):
        """Compute loss of a mutil scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (tuple[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = [], [], [], []
        for i in range(len(anchor_list)):
            anchors = anchor_list[i]
            cls_score = cls_scores[i]
            bbox_pred = bbox_preds[i]
            labels = labels_list[i]
            label_weights = label_weights_list[i]
            bbox_targets = bbox_targets_list[i]
            alignment_metrics = alignment_metrics_list[i]
            stride = strides[i]

            assert stride[0] == stride[1], 'h stride is not equal to w stride!'
            anchors = anchors.reshape(-1, 4)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(
                -1, self.config.class_num)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            bbox_targets = bbox_targets.reshape(-1, 4)
            labels = labels.reshape(-1)
            alignment_metrics = alignment_metrics.reshape(-1)
            label_weights = label_weights.reshape(-1)
            targets = labels if self.epoch < self.config.initial_epoch else (
                labels, alignment_metrics)
            cls_loss_func = self.initial_loss_cls \
                if self.epoch < self.config.initial_epoch else self.loss_cls

            loss_cls = cls_loss_func(
                cls_score, targets, label_weights, avg_factor=1.0)

            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            bg_class_ind = self.config.class_num

            if labels.min() < bg_class_ind:
                pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)  ##########################

                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_pred = bbox_pred[pos_inds]
                pos_anchors = anchors[pos_inds]

                pos_decode_bbox_pred = pos_bbox_pred
                pos_decode_bbox_targets = pos_bbox_targets / stride[0]

                # regression loss
                pos_bbox_weight = self.centerness_target(pos_anchors, pos_bbox_targets) \
                    if self.epoch < self.config.initial_epoch \
                    else alignment_metrics[pos_inds]

                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)
            else:
                loss_bbox = bbox_pred.sum() * 0
                pos_bbox_weight = ms.Tensor(0., dtype=bbox_targets.dtype)

            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)
            cls_avg_factors.append(alignment_metrics.sum())
            bbox_avg_factors.append(pos_bbox_weight.sum())

        return losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors

    def centerness_target(self, anchors, gts):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = ms.ops.stack([l_, r_], axis=1)
        top_bottom = ms.ops.stack([t_, b_], axis=1)
        centerness = ms.ops.sqrt(
            (left_right.min(axis=-1) / left_right.max(axis=-1)) *
            (top_bottom.min(axis=-1) / top_bottom.max(axis=-1)))
        assert not ms.ops.isnan(centerness).any()
        return centerness

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = ms.ops.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

class TrainOneStepCell(nn.Cell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        reduce_flag (bool): The reduce flag. Default value is False.
        mean (bool): Allreduce method. Default value is False.
        degree (int): Device number. Default value is None.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, input_imgs, input_boxes, input_classes):
        loss = self.network(input_imgs, input_boxes, input_classes)
        # print(f"loss_cls:{sum(loss['loss_cls']).asnumpy():.5f}  loss_bbox:{sum(loss['loss_bbox']).asnumpy():.5f} ")

        # loss = sum(loss['loss_cls']) + sum(loss['loss_bbox'])
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(input_imgs, input_boxes, input_classes, sens)
        grads = C.clip_by_global_norm(grads, clip_norm=3.0)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
