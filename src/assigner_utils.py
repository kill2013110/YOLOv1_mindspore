import mindspore as ms
from functools import partial

# def multi_apply(func, *args, **kwargs):
#     """Apply function to a list of arguments.
#
#     Note:
#         This function applies the ``func`` to multiple inputs and
#         map the multiple outputs of the ``func`` into different
#         list. Each list contains the same type of outputs corresponding
#         to different inputs.
#
#     Args:
#         func (Function): A function that will be applied to a list of
#             arguments
#
#     Returns:
#         tuple(list): A tuple containing multiple list, each list contains \
#             a kind of returned results by the function
#     """
#     pfunc = partial(func, **kwargs) if kwargs else func
#     map_results = map(pfunc, *args)
#     return tuple(map(list, zip(*map_results)))


# def unmap(data, count, inds, fill=0):
#     """Unmap a subset of item (data) back to the original set of items (of size
#     count)"""
#     if data.dim() == 1:
#         ret = data.new_full((count, ), fill)
#         ret[inds.type(ms.Tensor.bool)] = data
#     else:
#         new_size = (count, ) + data.shape[1:]
#         ret = data.new_full(new_size, fill)
#         ret[inds.type(ms.Tensor.bool), :] = data
#     return ret
#
# def anchor_inside_flags(flat_anchors,
#                         valid_flags,
#                         img_shape,
#                         allowed_border=0):
#     """Check whether the anchors are inside the border.
#
#     Args:
#         flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
#         valid_flags (torch.Tensor): An existing valid flags of anchors.
#         img_shape (tuple(int)): Shape of current image.
#         allowed_border (int, optional): The border to allow the valid anchor.
#             Defaults to 0.
#
#     Returns:
#         ms.Tensor: Flags indicating whether the anchors are inside a \
#             valid range.
#     """
#     img_h, img_w = img_shape[:2]
#     if allowed_border >= 0:
#         inside_flags = valid_flags & \
#             (flat_anchors[:, 0] >= -allowed_border) & \
#             (flat_anchors[:, 1] >= -allowed_border) & \
#             (flat_anchors[:, 2] < img_w + allowed_border) & \
#             (flat_anchors[:, 3] < img_h + allowed_border)
#     else:
#         inside_flags = valid_flags
#     return inside_flags

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    # Example:
    #     >>> bboxes1 = torch.FloatTensor([
    #     >>>     [0, 0, 10, 10],
    #     >>>     [10, 10, 20, 20],
    #     >>>     [32, 32, 38, 42],
    #     >>> ])
    #     >>> bboxes2 = torch.FloatTensor([
    #     >>>     [0, 0, 10, 20],
    #     >>>     [0, 10, 10, 19],
    #     >>>     [10, 10, 20, 20],
    #     >>> ])
    #     >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
    #     >>> assert overlaps.shape == (3, 3)
    #     >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
    #     >>> assert overlaps.shape == (3, )
    #
    # Example:
    #     >>> empty = torch.empty(0, 4)
    #     >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
    #     >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
    #     >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
    #     >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.shape[1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = ms.ops.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = ms.ops.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        # wh = fp16_clamp(rb - lt, min=0)
        wh = (rb - lt).clamp(0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = ms.ops.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = ms.ops.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = ms.ops.maximum(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = ms.ops.minimum(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = ms.ops.minimum(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = ms.ops.maximum(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = ms.Tensor([eps],dtype=union.dtype)
    union = ms.ops.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    # enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(0)

    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = ms.ops.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def sample(assign_result, flat_anchors, gt_bboxes):
    bboxes = flat_anchors
    (num_gt, assigned_gt_inds, max_overlaps, assigned_labels, pos_inds, neg_inds, assigned_labels) = assign_result

    gt_flags = ms.ops.zeros(bboxes.shape[0], dtype=ms.uint8)

    pos_bboxes = bboxes[pos_inds]
    neg_bboxes = bboxes[neg_inds]
    pos_is_gt = gt_flags[pos_inds]

    num_gts = gt_bboxes.shape[0]
    pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1

    if gt_bboxes.numel() == 0:
        # hack for index error case
        assert pos_assigned_gt_inds.numel() == 0
        pos_gt_bboxes = ms.empty_like(gt_bboxes).view(-1, 4)
    else:
        if len(gt_bboxes.shape) < 2:
            gt_bboxes = gt_bboxes.view(-1, 4)

        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

    if assigned_labels is not None:
        pos_gt_labels = assigned_labels[pos_inds]
    else:
        pos_gt_labels = None

    sample_restlt = (pos_inds, neg_inds, pos_bboxes, neg_bboxes, pos_is_gt, num_gts, pos_assigned_gt_inds, pos_gt_bboxes, pos_gt_labels)
    return sample_restlt

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = ms.ops.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets
