# pytorch implementation of filterbank pyramid filter
import cv2
import torch
import torch.nn.functional as F


class Spyr_PyTorch(object):

    def __init__(self, filter, device, height=5, nbands=4, sub_sample=True):
        '''
        :param filter: a function which returns filter parameters
        :param height:
        :param nbands:
        :param sub_sample: it shoule be Ture, haven't implemented the non-downsampling version
        :param device:
        :param wsize: window size
        '''
        params = vars()
        del params['self']
        self.__dict__.update(params)

    def build(self, img):
        '''
        :param img [N,C=1,H,W]
        :return:
        '''
        hi0filt = self.filter['hi0filt']
        lo0filt = self.filter['lo0filt']

        hi0 = self._conv2d(img, hi0filt)
        lo0 = self._conv2d(img, lo0filt)

        return [hi0] + self._buildLevs(lo0, self.height-1)

    def _buildLevs(self, lo0, height):
        if height<=1:
            return [lo0]

        coeffs = []
        bfilts = self.filter['bfilts']
        for ori in range(self.nbands):
            coeffs.append(self._conv2d(lo0, bfilts[ori]))

        lo = self._conv2d(lo0, self.filter['lofilt'])
        if self.sub_sample:    # sub-sampling
            lo = lo[:,:,::2,::2]    # same as F.interpolate

        return [coeffs] + self._buildLevs(lo, height-1)

    def _conv2d(self, img, kernel):
        # circular padding
        pad = kernel.shape[-1]//2
        img = torch.cat([img, img[:,:, 0:pad,:]], dim=-2)
        img = torch.cat([img, img[:,:,:, 0:pad]], dim=-1)
        img = torch.cat([img[:,:, -2 * pad:-pad,:], img], dim=-2)
        img = torch.cat([img[:,:,:, -2 * pad:-pad], img], dim=-1)

        # F.conv2d
        return F.conv2d(img, kernel)

    def getlist(self, coeff):
        straight = [bands for scale in coeff[1:-1] for bands in scale]
        straight = [coeff[0]] + straight + [coeff[-1]]
        return straight


if __name__ == "__main__":
    device = torch.device('cuda:0')
    from sp3Filters import sp3Filters
    s = Spyr_PyTorch(sp3Filters, sub_sample=True, device = device)

    image_path = '../data/0.png'
    img = cv2.imread(image_path, 0)
    img = torch.tensor(img).to(device)
    img = img.unsqueeze(0).unsqueeze(0).float()/255

    import pdb;
    pdb.set_trace()
    coeffs = s.build(img.double())
