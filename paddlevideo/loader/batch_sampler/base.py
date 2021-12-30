from paddle.fluid.dataloader import BatchSampler


class BaseBatchSampler(BatchSampler):
    def __init__(self, dataset=None, sapmler=None, **kwargs):
        super(BaseBatchSampler, self).__init__(dataset, sapmler, **kwargs)
