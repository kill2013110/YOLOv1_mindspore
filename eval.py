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
"""TOOD EVAL"""
import json
import os
import argparse
import cv2
import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.ops as ops

from mindspore import Tensor
from mindspore import context
from mindspore.ops import stop_gradient

from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.yolov1 import YOLOv1Detector
from src.eval_utils import post_process
from src.eval_utils import ClipBoxes
from src.augment import rescale_size
import copy

import torch




CLASSES_NAME = (
    '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush')

class COCOGenerator:

    def __init__(self, dataset_dir, annotation_file, resize_size):
        self.coco = COCO(annotation_file)
        self.root = dataset_dir
        ids = list(sorted(self.coco.imgs.keys()))
        ids = ids[:1]
        # ids = ids[1:2]
        # print("INFO====>check annos, filtering invalid data......")
        # new_ids = []
        # empty_ids = []
        # for i in ids:
        #     ann_id = self.coco.getAnnIds(imgIds=i, iscrowd=None)
        #     ann = self.coco.loadAnns(ann_id)
        #     if self._has_valid_annotation(ann):
        #         new_ids.append(i)
        #     else:
        #         empty_ids.append(i)

        new_ids = ids
        self.ids = new_ids

        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.resize_size = resize_size

        # self.mean = [0.40789654, 0.44719302, 0.47026115]
        # self.std = [0.28863828, 0.27408164, 0.27809835]
        '''TOOD official config'''
        self.mean =[123.675, 116.28,  103.53]
        self.std = [58.395, 57.12,  57.375]
        self.max_num = 100
        self.divisor = 32
        # (mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    def getImg(self, index):
        img_id = self.ids[index]
        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        # img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = cv2.imread(os.path.join(self.root, path), 1)

        return img, target

    def __getitem__(self, index):
        '''
        MMDetection TOOD eval data transforms:

        [Resize(img_scale=None, multiscale_mode=range, ratio_range=None, keep_ratio=True, bbox_clip_border=True),
        RandomFlip(flip_ratio=None),
        Normalize(mean=[123.675 116.28  103.53 ], std=[58.395 57.12  57.375], to_rgb=True),
        Pad(size=None, size_divisor=32, pad_to_square=False, pad_val={'img': 0, 'masks': 0, 'seg': 255}),
        ImageToTensor(keys=['img']),
        ollect(keys=['img'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'))]
        '''

        img, ann = self.getImg(index)
        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        # xywh2xyxy
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]
        img = np.array(img)

        '''Resize'''
        img, boxes, scale_factor = self.preprocess_img_boxes(img, boxes, self.resize_size)

        # img = np.flip(img, axis=1)
        w = img.shape[1]
        if boxes.shape[0] != 0:
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 2] = xmax
            boxes[:, 0] = xmin

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]


        '''Normalize'''
        img = img[:, :, ::-1]  # bgr2rgb
        normalize_op = ms.dataset.vision.Normalize(mean=self.mean, std=self.std)

        img = normalize_op(img)

        '''pad 32'''
        img_shape = img.shape
        pad_h = int(np.ceil(img_shape[0] / self.divisor)) * self.divisor
        pad_w = int(np.ceil(img_shape[1] / self.divisor)) * self.divisor
        width = max(pad_w - img_shape[1], 0)
        height = max(pad_h - img_shape[0], 0)
        padding = (0, 0, width, height)
        img = cv2.copyMakeBorder(img, padding[1], padding[3], padding[0], padding[2],
                                            cv2.BORDER_CONSTANT, value=0)

        '''ToTensor'''
        img = img.transpose(2, 0, 1)  # hwc2chw
        img = ms.Tensor(img)

        boxes = np.pad(boxes, ((0, max(self.max_num - len(boxes), 0)), (0, 0)), 'constant', constant_values=-1)
        classes = np.pad(classes, (0, max(self.max_num - len(classes), 0)), 'constant', constant_values=-1).astype('int32')
        box_info = {"boxes": ms.Tensor(boxes), "classes": ms.Tensor(classes), "scale_factor": ms.Tensor(scale_factor)}
        return img, box_info

    def __len__(self):
        return len(self.ids)
    def preprocess_img_boxes(self, img, boxes, scale):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        h, w = img.shape[:2]
        new_size, scale_factor = rescale_size((w, h), scale[::-1], return_scale=True)

        resized_img = cv2.resize(img, new_size, interpolation=1)
        # cv2_interp_codes {'nearest': 0, 'bilinear': 1, 'bicubic': 2, 'area': 3, 'lanczos': 4}
        new_h, new_w = resized_img.shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)

        if boxes is not None:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * w_scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * h_scale
        return resized_img, boxes, scale_factor

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if annot is None:
            return False
        if self._has_only_empty_bbox(annot):
            return False

        return True

# coordinate = np.int32(boxes[0,-10:].copy()*scale)
# cls = np.int32(labels[0,-10:].copy())
# # print(CLASSES_NAME[labels[0,-1]])
# # points = np.float32(padded_labels[:, 5:].copy())
# x = np.uint8(img[0].asnumpy().transpose(1,2,0).copy()+125)
# # x = np.uint8(img[0,0].asnumpy().copy()+100)
# for n in range(len(coordinate)):
#     cv2.rectangle(x, (coordinate[n][0], coordinate[n][1], \
#                       coordinate[n][2], coordinate[n][3]),
#                   [255, 255, 0], 2)
#     cv2.putText(x, f'{CLASSES_NAME[cls[n]]}',[coordinate[n][0], coordinate[n][1]], fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,255))
# cv2.imwrite('1.jpg', x)
def evaluate_coco(_generator, _model, threshold=0.05):
    results = []
    image_ids = []
    for index in tqdm(range(len(_generator))):
        img, box_info = _generator[index]
        scale_factor = box_info["scale_factor"]
        img = Tensor(img.copy(), ms.float32)
        expand_dims = ops.ExpandDims()
        img = expand_dims(img, 0)
        batch_imgs = img
        scores, labels, boxes = _model(img)
        scores, labels, boxes = post_process([scores, labels, boxes], 0.05, 0.6)
        boxes = ClipBoxes(batch_imgs, boxes)
        scores = stop_gradient(scores)
        labels = stop_gradient(labels)
        boxes = stop_gradient(boxes)
        boxes /= scale_factor

        # xyxy2xywh
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]
        boxes = boxes.asnumpy()
        labels = labels.asnumpy()
        scores = scores.asnumpy()
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < threshold:
                break
            image_result = {
                'image_id': _generator.ids[index],
                'category_id': _generator.id2category[label],
                'score': float(score),
                'bbox': box.tolist(),
            }
            results.append(image_result)
        image_ids.append(_generator.ids[index])
    json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)
    coco_true = _generator.coco
    coco_pred = coco_true.loadRes('coco_bbox_results.json')
    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", type=int, default=0, help="DEVICE_ID to run ")
parser.add_argument("--eval_path", type=str,
                    # default='/mnt/f/datasets/coco2017/images/train2017',
                    default='/mnt/f/datasets/coco2017/images/val2017',
                    )
parser.add_argument("--anno_path", type=str,
                    # default='/mnt/f/datasets/coco2017/annotations/instances_train2017.json',
                    default='/mnt/f/datasets/coco2017/annotations/instances_val2017.json',
                    )
parser.add_argument("--ckpt_path", type=str,
                    default="data1/TOOD/ckpt_0/ms8p_4-50_1.ckpt",
                    # default="tood_pth2ms_jit_manual_pad.ckpt",
                    # default="tood_pth2ms_jit.ckpt",
                    )

opt = parser.parse_args()
if __name__ == "__main__":

    # pth2ckpt()
    # context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=opt.device_id)
    context.set_context(
                        # mode=context.GRAPH_MODE,
                        mode=context.PYNATIVE_MODE,
                        device_target='GPU', device_id=opt.device_id)
    model = YOLOv1Detector(mode="inference")
    # model = TOODDetector(mode="inference")
    generator = COCOGenerator(opt.eval_path, opt.anno_path, [800, 1333])
    # generator = COCOGenerator(opt.eval_path, opt.anno_path, [640, 640])
    model.set_train(False)
    ms.load_param_into_net(model, ms.load_checkpoint(opt.ckpt_path))
    evaluate_coco(generator, model)


