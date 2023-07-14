# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mindspore as ms
from mindspore import nn

from .assigner_utils import bbox_overlaps

INF = 100000000

class ATSSAssigner():
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    If ``alpha`` is not None, it means that the dynamic cost
    ATSSAssigner is adopted, which is currently only used in the DDOD.

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk=9,
                 alpha=None,
                 iou_calculator=bbox_overlaps,
                 ignore_iof_thr=-1):
        self.topk = topk
        self.alpha = alpha
        self.iou_calculator = iou_calculator
        self.ignore_iof_thr = ignore_iof_thr

    """Assign a corresponding gt bbox or background to each bbox.

    Args:
        topk (int): number of bbox selected in each level.
        alpha (float): param of cost rate for each proposal only in DDOD.
            Default None.
        iou_calculator (dict): builder of IoU calculator.
            Default dict(type='BboxOverlaps2D').
        ignore_iof_thr (int): whether ignore max overlaps or not.
            Default -1 (1 or -1).
    """

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               cls_scores=None,
               bbox_preds=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        If ``alpha`` is not None, and ``cls_scores`` and `bbox_preds`
        are not None, the overlaps calculation in the first step
        will also include dynamic cost, which is currently only used in
        the DDOD.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO. Default None.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes. Default None.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4. Default None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]

        message = 'Invalid alpha parameter because cls_scores or ' \
                  'bbox_preds are None. If you want to use the ' \
                  'cost-based ATSSAssigner,  please set cls_scores, ' \
                  'bbox_preds and self.alpha at the same time. '

        if self.alpha is None:
            # ATSSAssigner
            overlaps = self.iou_calculator(bboxes, gt_bboxes)
            if cls_scores is not None or bbox_preds is not None:
                warnings.warn(message)
        else:
            # Dynamic cost ATSSAssigner in DDOD
            assert cls_scores is not None and bbox_preds is not None, message

            # compute cls cost for bbox and GT
            cls_cost = ms.ops.sigmoid(cls_scores[:, gt_labels])

            # compute iou between all bbox and gt
            overlaps = self.iou_calculator(bbox_preds, gt_bboxes)

            # make sure that we are in element-wise multiplication
            assert cls_cost.shape == overlaps.shape

            # overlaps is actually a cost matrix
            overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha

        # assign 0 by default
        assigned_gt_inds = ms.ops.zeros((num_bboxes, ), dtype=ms.int32)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = ms.ops.full((num_bboxes, ), -1, dtype=ms.int16)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = ms.ops.stack((gt_cx, gt_cy), axis=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = ms.ops.stack((bboxes_cx, bboxes_cy), axis=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)

            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = ms.ops.cat(candidate_idxs, axis=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, ms.ops.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        # ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
        #     num_gt, num_bboxes).contiguous().view(-1)
        # ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
        #     num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cx = ms.ops.repeat_interleave(bboxes_cx[None, :], num_gt, 0).view(-1)
        ep_bboxes_cy = ms.ops.repeat_interleave(bboxes_cy[None, :], num_gt, 0).view(-1)

        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = ms.ops.stack([l_, t_, r_, b_], axis=1).min(axis=1) > 0.01

        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = ms.ops.full_like(overlaps,-INF).t().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps = overlaps_inf.max(axis=1)
        argmax_overlaps = overlaps_inf.argmax(axis=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = ms.ops.full((num_bboxes, ), -1, dtype=assigned_gt_inds.dtype)
            pos_inds = ms.ops.nonzero(
                assigned_gt_inds > 0).squeeze()#########################################################################  Reshape warning???
            neg_inds = ms.ops.nonzero(
                assigned_gt_inds == 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        # return AssignResult(
        #     num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        return (num_gt, assigned_gt_inds, max_overlaps, assigned_labels, pos_inds, neg_inds, assigned_labels)


class TaskAlignedAssigner():
    """Task aligned assigner used in the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.

    Assign a corresponding gt bbox or background to each predicted bbox.
    Each bbox will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (int): number of bbox selected in each level
        iou_calculator (dict): Config dict for iou calculator.
            Default: dict(type='BboxOverlaps2D')
    """

    def __init__(self, topk=13, iou_calculator=bbox_overlaps):
        assert topk >= 1
        self.topk = topk
        self.iou_calculator = iou_calculator

    def assign(self,
               pred_scores,
               decode_bboxes,
               anchors,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               alpha=1.,
               beta=6.):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)


        Args:
            pred_scores (Tensor): predicted class probability,
                shape(n, num_classes)
            decode_bboxes (Tensor): predicted bounding boxes, shape(n, 4)
            anchors (Tensor): pre-defined anchors, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`TaskAlignedAssignResult`: The assign result.
        """
        anchors = anchors[:, :4]
        num_gt, num_bboxes = gt_bboxes.shape[0], anchors.shape[0]
        # compute alignment metric between all bbox and gt
        overlaps = self.iou_calculator(decode_bboxes, gt_bboxes)
        bbox_scores = pred_scores[:, gt_labels]
        # assign 0 by default
        assigned_gt_inds = ms.ops.zeros((num_bboxes, ), dtype=ms.int64)
        assign_metrics = ms.ops.zeros((num_bboxes,), dtype=ms.float32)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = ms.ops.zeros((num_bboxes,), dtype=ms.float32)
            if num_gt == 0:
                # No gt boxes, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = ms.ops.full((num_bboxes, ), -1, dtype=ms.int64)
            assign_result = AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            assign_result.assign_metrics = assign_metrics
            return assign_result

        # select top-k bboxes as candidates for each gt
        alignment_metrics = bbox_scores**alpha * overlaps**beta
        topk = min(self.topk, alignment_metrics.shape[0])
        _, candidate_idxs = alignment_metrics.topk(topk, dim=0, largest=True)
        candidate_metrics = alignment_metrics[candidate_idxs, ms.ops.arange(num_gt)]
        is_pos = candidate_metrics > 0

        # limit the positive sample's center in gt
        anchors_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
        anchors_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        ep_anchors_cx = ms.ops.repeat_interleave(anchors_cx[None, :], num_gt, 0).view(-1)
        ep_anchors_cy = ms.ops.repeat_interleave(anchors_cy[None, :], num_gt, 0).view(-1)

        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_anchors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_anchors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_anchors_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_anchors_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = ms.ops.stack([l_, t_, r_, b_], axis=1).min(axis=1) > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        overlaps_inf = ms.ops.full(overlaps.shape, -INF, dtype=overlaps.dtype).view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps = overlaps_inf.max(axis=1)
        argmax_overlaps = overlaps_inf.argmax(axis=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        assign_metrics[max_overlaps != -INF] = alignment_metrics[
            max_overlaps != -INF, argmax_overlaps[max_overlaps != -INF]]

        if gt_labels is not None:
            assigned_labels = ms.ops.full((num_bboxes, ), -1, dtype=assigned_gt_inds.dtype)
            pos_inds = ms.ops.nonzero(assigned_gt_inds > 0).squeeze()
            neg_inds = ms.ops.nonzero(assigned_gt_inds == 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        # assign_result = AssignResult(
        #     num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        # assign_result.assign_metrics = assign_metrics
        return (num_gt, assigned_gt_inds, max_overlaps, assigned_labels, pos_inds, neg_inds, assigned_labels,assign_metrics)