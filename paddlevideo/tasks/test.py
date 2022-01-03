# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import paddle
from paddlevideo.utils import get_logger, load
from ..loader.builder import build_dataloader, build_dataset, build_custom_dataloader, build_sampler
from ..metrics import build_metric
from ..modeling.builder import build_model
from ..modeling.framework import ManetSegment_Stage1
from ..modeling.framework.segment import ManetSegment_Stage2

logger = get_logger("paddlevideo")


@paddle.no_grad()
def test_model(cfg, weights, parallel=True):
    """Test model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path to load.
        parallel (bool): Whether to do multi-cards testing. Default: True.

    """
    if cfg.MODEL.framework == "ManetSegment_Stage2":
        cfg_helper = {"knns": 1, "is_save_image": True}
        cfg.update(cfg_helper)
        ManetSegment_Stage2().test_step(**cfg, weights=weights, parallel=False)
        return

    # 1. Construct model.
    if cfg.MODEL.get('backbone') and cfg.MODEL.backbone.get('pretrained'):
        cfg.MODEL.backbone.pretrained = ''  # disable pretrain model init
    model = build_model(cfg.MODEL)

    if parallel:
        model = paddle.DataParallel(model)

    # 2. Construct dataset and sampler.
    cfg.DATASET.test.test_mode = True
    dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
    cfg.dataset = dataset
    batch_size = cfg.DATASET.get("test_batch_size", 8)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    dataloader_setting = dict(batch_size=batch_size,
                              num_workers=num_workers,
                              places=places,
                              drop_last=False,
                              shuffle=False)

    if cfg.get('DATALOADER') and cfg.get('DATALOADER').get('test') and cfg.get(
            'DATALOADER').get('test').get('sampler'):
        sampler = cfg['DATALOADER']['test']['sampler']
        sampler['dataset'] = dataset
        sampler = build_sampler(sampler)
        cfg['DATALOADER']['test'].pop('sampler')
        dataloader_setting['sampler'] = sampler
    dataloader_setting.update({'dataset': dataset})
    data_loader = None
    if cfg.get('DATALOADER') and cfg.get('DATALOADER').get('test'):
        dataloader_setting.update(cfg['DATALOADER']['test'])
        if cfg['DATALOADER']['test'].get('name'):
            data_loader = build_custom_dataloader(**dataloader_setting)
    if not data_loader:
        data_loader = build_dataloader(**dataloader_setting)
    model.eval()

    state_dicts = load(weights)
    model.set_state_dict(state_dicts)

    # add params to metrics
    cfg.METRIC.data_size = len(dataset)
    cfg.METRIC.batch_size = batch_size

    Metric = build_metric(cfg.METRIC)
    for batch_id, data in enumerate(data_loader):
        outputs = model(data, mode='test')
        Metric.update(batch_id, data, outputs)
    Metric.accumulate()
