# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path as osp
import copy
import numpy as np
from abc import ABC, abstractmethod
from paddle.io import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 feed_list=None,
                 places=None,
                 return_list=True,
                 batch_sampler=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 collate_fn=None,
                 num_workers=0,
                 use_buffer_reader=True,
                 use_shared_memory=True,
                 timeout=0,
                 worker_init_fn=None,
                 persistent_workers=False):
        super.__init__(dataset=dataset,
                       feed_list=feed_list,
                       places=places,
                       return_list=return_list,
                       batch_sampler=batch_sampler,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       drop_last=drop_last,
                       collate_fn=collate_fn,
                       num_workers=num_workers,
                       use_buffer_reader=use_buffer_reader,
                       use_shared_memory=use_shared_memory,
                       timeout=timeout,
                       worker_init_fn=worker_init_fn,
                       persistent_workers=persistent_workers)
