from paddle.fluid.reader import DataLoader

from paddlevideo.loader.dataloader.base import BaseDataLoader
from paddlevideo.loader.registry import DATALOADER


@DATALOADER.register()
class Stage2_DataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(**kwargs)
