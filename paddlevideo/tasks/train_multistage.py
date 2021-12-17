from paddlevideo.tasks import train_model


def cfg_parser(cfg, key, all=['stage1']):
    if isinstance(cfg, dict):
        ret = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                v = cfg_parser(v, key, all)
            elif isinstance(v, list):
                v_ = []
                for l in v:
                    v_.append(cfg_parser(l, key, all))
                v = v_
            if k == key:
                ret.update(v)
            elif k not in all:
                ret[k] = v
        return ret
    elif isinstance(cfg, list):
        ret = []
        for l in cfg:
            ret.append(cfg_parser(l, key, all=['stage1']))
        return ret
    else:
        return cfg


def train_model_multistage(cfg, **kwargs):
    for k in cfg.STAGE:
        cfg_stage = cfg_parser(cfg, k, cfg.STAGE)
        train_model(cfg_stage, **kwargs)


if __name__ == '__main__':
    import os

    import yaml

    fs = open("/configs/segmentationer/manet_stage1.yaml", encoding="UTF-8")
    datas = yaml.load(fs, Loader=yaml.FullLoader)
    print(datas)

    cfg_stage1, cfg_stage2 = cfg_parser(datas, key=['stage1', 'stage2'])
    assert cfg_stage1 != cfg_stage2
    print(cfg_stage1)
    print(cfg_stage2)
