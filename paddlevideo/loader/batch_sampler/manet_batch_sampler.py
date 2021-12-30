from __future__ import absolute_import
from collections import defaultdict
import numpy as np

from .base import BaseBatchSampler
from ..registry import BATCH_SAMPLERS


@BATCH_SAMPLERS.register()
class ManetBatchSampler_stage2(BaseBatchSampler):
    def __init__(self, **kwargs):
        super(ManetBatchSampler_stage2, self).__init__(**kwargs)
