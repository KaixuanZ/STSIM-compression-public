# This is a pytoch implementation of DISTS metric.
# Requirements: python >= 3.6, pytorch >= 1.0

import numpy as np
import os,sys
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class VGG(torch.nn.Module):
    def __init__(self, weights_path=None, numpy=False):
        super(VGG, self).__init__()
        self.numpy = numpy

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        #return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, require_grad=False, batch_average=False):
        if x.shape[2]!=256 or x.shape[3]!=256:
            x = F.interpolate(x, size=(256,256))
        if require_grad:
            feats = self.forward_once(x)
        else:
            with torch.no_grad():
                feats = self.forward_once(x)
        return feats

    def build(self, x, require_grad=False, batch_average=False):
        res = self.forward(x, require_grad, batch_average)
        return res

    def getlist(self, coeffs):
        res = []
        if self.numpy:
            for layer in coeffs:
                # remove the C dim
                res += torch.unbind(layer,1)
            # remove N dim
            res = [r[0].detach().cpu().numpy() for r in res]
        else:
            for layer in coeffs:
                res += torch.split(layer,1,1)
        return res

def prepare_image(image, resize=True):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from utils.dataset_concatenated import Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    dataset_dir = '../concatenated'  # config['dataset_dir']
    label_file = 'label.xlsx'
    dist = 'test'
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset))

    X1, X2, Y, mask = next(iter(test_loader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG().to(device)

    ref = X1.to(device)
    dist = X2.to(device)
    feat1 = model(ref[:1])
    feat2 = model(dist[:1])

    tmp1 = model.getlist(feat1)
    import pdb;
    pdb.set_trace()