# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

from paddlevideo.loader.dataset import DAVIS2017_Feature_ExtractDataset, DAVIS2017_VOS_TrainDataset, \
    DAVIS2017_TrainDataset
from paddlevideo.loader.pipelines import custom_transforms_f as tr, RandomHorizontalFlip_manet, Resize_manet, \
    ToTensor_manet, RandomCrop_manet, RandomScale_manet

import os
import time
import timeit
import davisinteractive.robot.interactive_robot as interactive_robot
import cv2
import numpy as np
import paddle
from PIL.Image import Image
from davisinteractive.session import DavisInteractiveSession
from davisinteractive.utils.scribbles import scribbles2mask, annotated_frames
from paddle import nn
from paddlevideo.utils.manet_utils import float_, _palette, damage_masks, int_, long_, label2colormap, mask_damager, \
    byte_
from ... import builder
from ...registry import SEGMENTATIONERS
from .base import BaseSegmentationer


@SEGMENTATIONERS.register()
class ManetSegmentationer_Stage1(BaseSegmentationer):
    def __init__(self, backbone=None, head=None, **cfg):
        super().__init__(backbone, **cfg)
        head_copy = head.copy()
        head_copy.update({'feature_extracter': self.backbone})
        self.head = builder.build_head(head_copy)

    def train_step(self, data_batch, step, **cfg):
        """Define how the model is going to train, from input to output.
        返回任何你想打印到日志中的东西
        """
        ref_imgs = data_batch['ref_img']  # batch_size * 3 * h * w
        img1s = data_batch['img1']
        img2s = data_batch['img2']
        ref_scribble_labels = data_batch[
            'ref_scribble_label']  # batch_size * 1 * h * w
        label1s = data_batch['label1']
        label2s = data_batch['label2']
        seq_names = data_batch['meta']['seq_name']
        obj_nums = data_batch['meta']['obj_num']

        bs, _, h, w = img2s.shape
        inputs = paddle.concat((ref_imgs, img1s, img2s), 0)
        if self.cfg['TRAIN']['damage_initial_previous_frame_mask']:
            try:
                label1s = damage_masks(label1s)
            except:
                label1s = label1s
                print('damage_error')

        tmp_dic = self.head(inputs,
                            ref_scribble_labels,
                            label1s,
                            use_local_map=True,
                            seq_names=seq_names,
                            gt_ids=obj_nums,
                            k_nearest_neighbors=self.cfg['TRAIN']['knns'])
        label_and_obj_dic = {}
        for i, seq_ in enumerate(seq_names):
            label_and_obj_dic[seq_] = (label2s[i], obj_nums[i])
        for seq_ in tmp_dic.keys():
            tmp_pred_logits = tmp_dic[seq_]
            tmp_pred_logits = nn.functional.interpolate(tmp_pred_logits,
                                                        size=(h, w),
                                                        mode='bilinear',
                                                        align_corners=True)
            tmp_dic[seq_] = tmp_pred_logits

            label_tmp, obj_num = label_and_obj_dic[seq_]
            obj_ids = np.arange(1, obj_num + 1)
            obj_ids = paddle.to_tensor(obj_ids)
            obj_ids = int_(obj_ids)

        loss_metrics = {
            'loss':
            self.head.loss() / bs
            # self.head.loss(tmp_dic, label_tmp, step, obj_ids=obj_ids, seq_=seq_) / bs
        }
        return loss_metrics

    def val_step(self, data_batch, **kwargs):
        pass

    def infer_step(self, data_batch, **kwargs):
        """Define how the model is going to test, from input to output."""
        pass
