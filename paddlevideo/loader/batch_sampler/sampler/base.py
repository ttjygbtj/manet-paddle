from paddle.fluid.dataloader import Sampler


class BaseSampler(Sampler):
    def __init__(self, data_source):
        super(BaseSampler, self).__init__(data_source)

