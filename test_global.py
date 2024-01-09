import argparse
import numpy as np
from utils.dataset import Dataset
from utils.dataset import Dataset_TIP_all
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

import matplotlib.pyplot as plt
def plot(X, Y, mask, figname='tmp'):
    X = X.squeeze(-1)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    y = X
    x = Y
    for i in set(mask):
        plt.scatter(x[mask==i],y[mask==i], label='cls'+str(int(i)))
    plt.xlabel('label')
    plt.ylabel('prediction')

    PCoeff = scipy.stats.pearsonr(x,y)[0]
    SCoeff = scipy.stats.spearmanr(x, y).correlation
    KCoeff = scipy.stats.kendalltau(x, y).correlation
    plt.legend()
    plt.title("PLCC={:.3f},SRCC={:.3f},KRCC={:.3f}".format(PCoeff,SCoeff,KCoeff))
    plt.savefig(figname + '.eps')
    plt.close()

def plot2(X1, X2, Y, mask, figname='tmp'):
    X1 = X1.squeeze(-1)
    X1 = X1.detach().cpu().numpy()
    X2 = X2.squeeze(-1)
    X2 = X2.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    for i in set(mask):
        x1 = X1[mask==i]
        x2 = X2[mask==i]
        y = Y[mask==i]
        plt.scatter(x1,y,color='r')
        plt.scatter(x2,y,color='b')
        plt.xlabel('prediction')
        plt.ylabel('label')
        plt.savefig('_'.join([figname,'class'+str(int(i))+'.png']))
        plt.close()

def save_as_np(pred, Y):
    path = 'STSIM-M-global'
    # os.mkdir(path)
    np.save(os.path.join(path,'pred_M.npy'), pred.detach().cpu().numpy())
    np.save(os.path.join(path,'label.npy'), Y.detach().cpu().numpy())    # label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="config/test_DISTS_global.cfg", help="path to data config file")
    parser.add_argument("--config", type=str, default="config/train_STSIM_global.cfg", help="path to data config file")
    # parser.add_argument("--config", type=str, default="config/test_STSIM_global.cfg", help="path to data config file")
    parser.add_argument("--n_textures", type=int, default=10, help="number of textures for training, 10 or 22")
    parser.add_argument("--batch_size", type=int, default=4080, help="size of each image batch")
    opt = parser.parse_args()
    print(opt)

    config = parse_config(opt.config)
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read train config
    import json
    with open(config['train_config_path']) as f:
        train_config = json.load(f)
        print(train_config)

    # read data
    dataset_dir = train_config['dataset_dir']
    label_file = train_config['label_file']
    dist = train_config['test']
    # testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    if opt.n_textures == 10:
        testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    elif opt.n_textures == 22:
        testset = Dataset_TIP_all(data_dir=dataset_dir, label_file=label_file, dist=dist)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)
    X1, X2, Y, mask, pt = next(iter(test_loader))

    # test with different model
    if config['model'] == 'PSNR':
        tmp = torch.tensor([torch.max(X1[i]) for i in range(X1.shape[0])])
        pred = 10 * torch.log10(tmp * tmp / torch.mean((X1 - X2) ** 2, dim=[1, 2, 3]))
        print("PSNR test:", evaluation(pred, Y))

    elif config['model'] == 'STSIM':
        from metrics.STSIM import *
        X1 = X1.to(device).double()
        X2 = X2.to(device).double()
        Y = Y.to(device).double()

        filter = train_config['filter']
        m_g = Metric(filter, device=device)

        # pred = m_g.STSIM1(X1, X2)
        # print("STSIM-1 test:", evaluation(pred, Y))
        #
        # pred = m_g.STSIM2(X1, X2)
        # print("STSIM-2 test:", evaluation(pred, Y))

        # path = train_config['weights_path'].split('/')
        # path[-1] = 'STSIM-M.pt'
        # weight_M = torch.load('/'.join(path))
        # pred = m_g.STSIM_M(X1, X2, weight=weight_M)
        # print("STSIM-M test:", evaluation(pred, Y))  #  {'PLCC': 0.874, 'SRCC': 0.834, 'KRCC': 0.73}
        # save_as_np(pred, Y)
        # # plot(pred,Y,mask,'STSIM-M_table3')
        #
        # path = train_config['weights_path'].split('/')
        # path[-1] = 'STSIM-I.pt'
        # weight_I = torch.load('/'.join(path))
        # pred = m_g.STSIM_I(X1, X2, weight=weight_I)
        # print("STSIM-I test:", evaluation(pred, Y))  #  {'PLCC': 0.894, 'SRCC': 0.852, 'KRCC': 0.736}
        # #plot(pred,Y,'STSIM-I')
        #

        model = STSIM_M(train_config['dim'], mode=int(train_config['mode']), filter = filter, device = device)

        model.load_state_dict(torch.load(train_config['weights_path']))

        model.to(device).double()
        if filter=='VGG':
            pred = []
            pred.append(model(X1[:45],X2[:45]))
            pred.append(model(X1[45:],X2[45:]))
            pred = torch.cat(pred)
        else:
            pred = model(X1, X2)
        # import pdb;pdb.set_trace()
        # plot(pred, Y, mask, 'metric_pred')
        print("STSIM-M (trained) test:", evaluation(pred, Y)) # for complex: {'PLCC': 0.983, 'SRCC': 0.979, 'KRCC': 0.944}

        # save_as_np(pred, Y)
        # plot2(pred_STSIM_2, pred, Y ,mask,'metric_pred')

    elif config['model'] == 'DISTS':
        test_loader = torch.utils.data.DataLoader(testset, batch_size=60)
        from metrics.DISTS_pt import *

        # import pdb;pdb.set_trace()
        model = DISTS(weights_path=train_config['weights_path']).to(device)
        pred, Y, mask = [], [], []
        for X1_test, X2_test, Y_test, _, _ in test_loader:
            X1_test = F.interpolate(X1_test, size=256).float().to(device)
            X2_test = F.interpolate(X2_test, size=256).float().to(device)
            Y_test = Y_test.to(device)
            pred.append(model(X1_test, X2_test))
            Y.append(Y_test)
        pred = torch.cat(pred, dim=0).detach()
        Y = torch.cat(Y, dim=0).detach()

        print("DISTS test:", evaluation(pred, Y))  # {'PLCC': 0.9574348579861184, 'SRCC': 0.9213941434033467, 'KRCC': 0.8539799877032255}