import argparse
import os
import numpy as np
from utils.dataset import Dataset
import torch
from metrics.STSIM import *
import scipy.stats

def SpearmanCoeff(X, Y, mask):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    coeff = 0
    N = set(mask.detach().cpu().numpy())
    X = X.squeeze(-1)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    for i in N:
        X1 = X[mask == i]
        X2 = Y[mask == i]

        coeff += np.abs(scipy.stats.spearmanr(X1, X2).correlation)
    return coeff / len(N)

def KendallCoeff(X, Y, mask):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    coeff = 0
    N = set(mask.detach().cpu().numpy())
    X = X.squeeze(-1)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    for i in N:
        X1 = X[mask == i]
        X2 = Y[mask == i]

        coeff += np.abs(scipy.stats.kendalltau(X1, X2).correlation)
    return coeff / len(N)

def PearsonCoeff(X, Y, mask):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    coeff = 0
    N = set(mask.detach().cpu().numpy())
    X = X.squeeze(-1)
    for i in N:
        X1 = X[mask == i].double()
        X1 = X1 - X1.mean()
        X2 = Y[mask == i].double()
        X2 = X2 - X2.mean()

        nom = torch.dot(X1, X2)
        denom = torch.sqrt(torch.sum(X1 ** 2) * torch.sum(X2 ** 2))

        coeff += torch.abs(nom / (denom + 1e-10))
    return coeff / len(N)

def evaluation(pred, Y, mask):
    res = {}

    PCoeff = PearsonCoeff(pred, Y, mask).item()
    res['PLCC'] = float("{:.3f}".format(PCoeff))

    SCoeff = SpearmanCoeff(pred, Y, mask)
    res['SRCC'] = float("{:.3f}".format(SCoeff))

    KCoeff = KendallCoeff(pred, Y, mask)
    res['KRCC'] = float("{:.3f}".format(KCoeff))
    return res

def load_data(dist='train'):
    # data loading
    dataset_dir = 'data'
    label_file = 'label_final.xlsx'
    shuffle = True
    train_batch_size = 2000
    trainset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=shuffle)
    return train_loader

def STSIM_legacy():
    weight_path = 'weights/STSIM_11142022'
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)
    # load data
    device = 'cuda'
    filter = 'SF'   # 'SF' steerable filter or 'SCF' steerable complex filter
    train_loader = load_data('train')
    # valid_loader = load_data('valid')
    test_loader = load_data('test')
    X1_train, X2_train, Y_train, mask_train, _ = next(iter(train_loader))
    X1_test, X2_test, Y_test, mask_test, _ = next(iter(test_loader))
    X1_train = X1_train.double().to(device)
    X2_train = X2_train.double().to(device)
    Y_train = Y_train.double().to(device)
    mask_train = mask_train.double().to(device)
    X1_test = X1_test.double().to(device)
    X2_test = X2_test.double().to(device)
    X2_test = X2_test.double().to(device)
    Y_test = Y_test.double().to(device)
    mask_test = mask_test.double().to(device)

    # -----------------STSIM features -------------------------
    m = Metric(filter, device)
    X1 = m.STSIM(X1_train)
    X2 = m.STSIM(X2_train)
    mask = mask_train


    X_train = [X1[mask==i][0:1] for i in set(mask.detach().cpu().numpy())]
    mask_I = [mask[mask==i][0:1]  for i in set(mask.detach().cpu().numpy())]
    X_train.append(X2)
    mask_I.append(mask)

    X_train = torch.cat(X_train)
    mask_I = torch.cat(mask_I)

    # ----------------train----------------------
    # STSIM-M
    weight_M = m.STSIM_M(X_train)
    torch.save(weight_M, os.path.join(weight_path,'STSIM-M.pt'))
    # STSIM-I
    weight_I = m.STSIM_I(X_train, mask = mask_I)
    torch.save(weight_I, os.path.join(weight_path,'STSIM-I.pt'))

    # -----------------test-----------------------
    # results won't be the same as reported in paper, because validation set not included
    pred = m.STSIM_M(X1_test, X2_test, weight=weight_M)
    print("STSIM-M test:", evaluation(pred, Y_test, mask_test))

    pred = m.STSIM_I(X1_test, X2_test, weight=weight_I)
    print("STSIM-I test:", evaluation(pred, Y_test, mask_test))  # {'PLCC': 0.894, 'SRCC': 0.852, 'KRCC': 0.736}

    pred = m.STSIM1(X1_test, X2_test)
    print("STSIM-1 test:", evaluation(pred, Y_test, mask_test))

    pred = m.STSIM2(X1_test, X2_test)
    print("STSIM-2 test:", evaluation(pred, Y_test, mask_test))

STSIM_legacy()



