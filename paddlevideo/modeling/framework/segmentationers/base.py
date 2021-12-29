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

from abc import abstractmethod
import paddle.nn as nn

from tools.utils import build_train_helper, build_test_helper, build_valid_helper
from ... import builder


class BaseSegmentationer(nn.Layer):
    """Base class for Segmentationer.
    All Segmentationerer should subclass it.
    All subclass should overwrite:
    - Methods:``train_step``, define your train step.
    - Methods:``valid_step``, define your valid step, always the same as train_step.
    - Methods:``test_step``, define your test step.
    """
    def __init__(self, backbone=None, head=None, **cfg):
        super().__init__()
        if backbone != None:
            self.backbone = builder.build_backbone(backbone)
            if hasattr(self.backbone, 'init_weights'):
                self.backbone.init_weights()
        else:
            self.backbone = None
        if head != None:
            self.head_name = head.name
            self.head = builder.build_head(head)
            if hasattr(self.head, 'init_weights'):
                self.head.init_weights()
        else:
            self.head = None
        self.cfg = cfg

    def init_weights(self):
        """Initialize the model network weights. """
        if getattr(self.backbone, 'init_weights'):
            self.backbone.init_weights()
        else:
            pass

    def forward(self, data_batch, mode='infer', **kwargs):
        """
        1. Define how the model is going to run, from input to output.
        2. Console of train, valid, test or infer step
        3. Set mode='infer' is used for saving inference model, refer to tools/export_model.py
        """
        if mode == 'train':
            return self.train_step(data_batch, **kwargs)
        elif mode == 'valid':
            return self.val_step(data_batch, **kwargs)
        elif mode == 'test':
            return self.test_step(data_batch, **kwargs)
        elif mode == 'infer':
            return self.infer_step(data_batch, **kwargs)
        else:
            raise NotImplementedError

    @abstractmethod
    def train_step(self, data_batch, **kwargs):
        """Training step.  input_data_batch -> loss_metric
        """
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data_batch, **kwargs):
        """Validating setp. input_data_batch -> loss_metric
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, data_batch, **kwargs):
        """Tets setp. to get acc in test data. input_data_batch -> output
        """
        raise NotImplementedError