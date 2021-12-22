from paddle.fluid.dataloader import Sampler


class BaseSampler(Sampler):
    def __init__(self):
        super(BaseSampler, self).__init__()

    def __iter__(self):
        raise NotImplementedError
