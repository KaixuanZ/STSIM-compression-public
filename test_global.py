import argparse
import numpy as np
from utils.dataset import Dataset
from utils.parse_config import parse_config

import torch
import torch.nn.functional as F
import scipy.stats
import os

def SpearmanCoeff(X, Y):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    X = X.squeeze(-1)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    return np.abs(scipy.stats.spearmanr(X, Y).correlation)

def KendallCoeff(X, Y):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    X = X.squeeze(-1)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    return np.abs(scipy.stats.kendalltau(X, Y).correlation)

def PearsonCoeff(X, Y):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    X = X.squeeze(-1)

    X1 = X.double()
    X1 = X1 - X1.mean()
    X2 = Y.double()
    X2 = X2 - X2.mean()

    nom = torch.dot(X1, X2)
    denom = torch.sqrt(torch.sum(X1 ** 2) * torch.sum(X2 ** 2))

    coeff = torch.abs(nom / (denom + 1e-10))
    return coeff

def evaluation(pred, Y):
    res = {}

    PCoeff = PearsonCoeff(pred, Y).item()
    res['PLCC'] = float("{:.3f}".format(PCoeff))

    SCoeff = SpearmanCoeff(pred, Y)
    res['SRCC'] = float("{:.3f}".format(SCoeff))

    KCoeff = KendallCoeff(pred, Y)
    res['KRCC'] = float("{:.3f}".format(KCoeff))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test_STSIM_global.cfg", help="path to data config file")
    parser.add_argument("--batch_size", type=int, default=4080, help="size of each image batch")
    opt = parser.parse_args()
    print(opt)

    config = parse_config(opt.config)
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    dataset_dir = config['dataset_dir']
    label_file = config['label_file']
    dist = config['dist']
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    # read train config
    import json
    with open(config['train_config_path']) as f:
        train_config = json.load(f)
        print(train_config)

    X1, X2, Y, mask, pt = next(iter(test_loader))


    from metrics.STSIM import *
    X1 = X1.to(device).double()
    X2 = X2.to(device).double()
    Y = Y.to(device).double()

    filter = train_config['filter']

    model = STSIM_M(train_config['dim'], mode=int(train_config['mode']), filter = filter, device = device)
    model.load_state_dict(torch.load(train_config['weights_path']))
    model.to(device).double()
    with torch.no_grad():
        pred = model(X1, X2)
    print("STSIM-M (trained) test:", evaluation(pred, Y)) # for complex: {'PLCC': 0.983, 'SRCC': 0.979, 'KRCC': 0.944}
