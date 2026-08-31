import torch

def get_gustavo(model="", role="pretrained", head=True)
    clf = torch.hub.load("H-IAAC/benchmarking-encoders-ssl-har", "lfr_ts2vec_ms", role="finetuned", head=True, trust_repo=True)
    return clf