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
"""COCO_dataset"""
import os
import random
import cv2
import numpy as np
import mindspore as ms
import mindspore.dataset as de
import mindspore.dataset.vision as py_vision
import mindspore.dataset.vision as c_vision

from pycocotools.coco import COCO
from PIL import Image

from .augment import rescale_size

def flip(img, boxes):
    img = np.flip(img, axis=1)
    w = img.shape[1]
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes


class COCODataset:
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

    def __init__(self, dataset_dir, annotation_file, resize_size=[800, 1333], is_train=True, transform=None):
        self.resize_size = resize_size
        self.coco = COCO(annotation_file)
        self.root = dataset_dir
        ids = list(sorted(self.coco.imgs.keys()))[:5]
        # ids = list(sorted(self.coco.imgs.keys()))

        print('********* Filter images without ground truths. *************')
        new_ids, empty_ids = [], []
        for i in ids:
            ann_id = self.coco.getAnnIds(imgIds=i, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                new_ids.append(i)
            else:
                empty_ids.append(i)
        print(f'**********  valid imgs: {len(new_ids)}  **************')
        self.ids = new_ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.transform = transform
        self.train = is_train

        self.mean = [123.675, 116.28,  103.53]
        self.std = [58.395, 57.12,  57.375]
        self.max_num = 100
        self.divisor = 32

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
        MMDetection TOOD Training transform:

        Compose(
        LoadImageFromFile(to_float32=False, color_type='color', channel_order='bgr', file_client_args={'backend': 'disk'})
        LoadAnnotations(with_bbox=True, with_label=True, with_mask=False, with_seg=False, poly2mask=True, poly2mask={'backend': 'disk'})
        Resize(img_scale=[(1333, 800)], multiscale_mode=range, ratio_range=None, keep_ratio=True, bbox_clip_border=True)
        RandomFlip(flip_ratio=0.5)
        Normalize(mean=[123.675 116.28  103.53 ], std=[58.395 57.12  57.375], to_rgb=True)
        Pad(size=None, size_divisor=32, pad_to_square=False, pad_val={'img': 0, 'masks': 0, 'seg': 255})
        DefaultFormatBundle(img_to_float=True)
        Collect(keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'))
        )
        '''
        img, ann = self.getImg(index)
        # img, ann = self.getImg(0)
        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        # xywh2xyxy
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]

        # print('org: ', img.shape, img[:3, :3, 0], self.ids[index])

        img, boxes, scale_factor = self.preprocess_img_boxes(img, boxes, self.resize_size)

        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)

            '''TOOD 无其他数据增强'''
            # if self.transform is not None:
            #     img, boxes = self.transform(img, boxes)

        classes = [o['category_id'] for o in ann]
        # classes = [self.category2id[c] for c in classes]
        classes = [self.category2id[c] - 1 for c in classes]  # 与mmdet的实现对齐

        '''Normalize'''
        img = img[:, :, ::-1]  # bgr2rgb
        normalize_op = ms.dataset.vision.Normalize(mean=self.mean, std=self.std)
        img = normalize_op(img)

        # print('Normalize: ', img.shape, img[:3,:3,0], self.ids[index])

        '''pad 32'''
        img_shape = img.shape
        pad_h = int(np.ceil(img_shape[0] / self.divisor)) * self.divisor
        pad_w = int(np.ceil(img_shape[1] / self.divisor)) * self.divisor
        width = max(pad_w - img_shape[1], 0)
        height = max(pad_h - img_shape[0], 0)
        padding = (0, 0, width, height)
        img = cv2.copyMakeBorder(img, padding[1], padding[3], padding[0], padding[2],
                                            cv2.BORDER_CONSTANT, value=0)
        # print('pad: ', img.shape, img[:3,:3,0], '\n')

        '''ToTensor'''
        img = img.transpose(2, 0, 1)
        # img = ms.Tensor(img) # 可以不转tensor

        boxes = np.pad(boxes, ((0, max(self.max_num - len(boxes), 0)), (0, 0)), 'constant', constant_values=-1)
        classes = np.pad(classes, (0, max(self.max_num - len(classes), 0)), 'constant', constant_values=-1).astype('int32')
        return img, boxes, classes

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

def create_coco_dataset(dataset_dir, annotation_file, batch_size, resize_size=None,shuffle=True, \
                        transform=None, num_parallel_workers=8, num_shards=None, shard_id=None):
    # cv2.setNumThreads(0)
    # dataset = COCODataset(dataset_dir, annotation_file, resize_size=resize_size, is_train=True, transform=transform)
    # dataset_column_names = ["img", "boxes", "class"]
    # ds = de.GeneratorDataset(dataset, column_names=dataset_column_names, shuffle=shuffle, num_parallel_workers=1, num_shards=num_shards, shard_id=shard_id)
    #
    # '''padded_batch 会自动pad对齐每张图片,等效_MapDatasetFetcher类中的collate_fn'''
    # ds = ds.padded_batch(batch_size, pad_info={}, num_parallel_workers=min(8, num_parallel_workers), drop_remainder=True)
    # return ds, len(dataset)

    cv2.setNumThreads(0)
    dataset = COCODataset(dataset_dir, annotation_file, is_train=True, transform=transform)
    dataset_column_names = ["img", "boxes", "class"]
    ds = de.GeneratorDataset(dataset, column_names=dataset_column_names, \
    shuffle=shuffle, num_parallel_workers=min(8, num_parallel_workers), num_shards=num_shards, shard_id=shard_id)
    ds = ds.padded_batch(batch_size, pad_info={}, num_parallel_workers=min(8, num_parallel_workers), drop_remainder=True)
    return ds, len(dataset)
