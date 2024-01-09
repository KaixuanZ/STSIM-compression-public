import torch
from scipy.fftpack import dct

class DCT(object):
    def __init__(self, blocksize=8, device=None):
        self.device = torch.device('cpu') if device is None else device
        self.blocksize = blocksize

    def build(self, imgs):
        res = []
        for i in range(imgs.shape[0]):
            img = imgs[i,0]
            img = img.cpu().numpy()
            coeffs = []
            for h in range(img.shape[0]//self.blocksize):
                for w in range(img.shape[1]//self.blocksize):
                    patch = img[h *self.blocksize:(h+1)*self.blocksize, w*self.blocksize:(w+1)*self.blocksize]
                    coeffs.append(torch.tensor(dct(patch, type=4)).to(self.device))
            res.append(torch.stack(coeffs))
        res = torch.stack(res)  # [N, H*W//blocksize//blocksize , blocksize, blocksize]

        coeffs = res.reshape([imgs.shape[0], imgs.shape[2]//self.blocksize, imgs.shape[3]//self.blocksize, -1])    #   [N, H//blocksize, W//blocksize, blocksize**2]
        res = []
        for i in range(self.blocksize):
            tmp = []
            for j in range(self.blocksize):
                tmp.append(coeffs[:,:,:,i*self.blocksize+j].unsqueeze(1))
            res.append(tmp)
        return res

    def getlist(self, coeffs):
        straight = [bands for scale in coeffs for bands in scale]
        return straight

if __name__ == "__main__":
    device = torch.device('cuda:0')
    fb = DCT(device = device)

    import cv2
    image_path = '../data/0.png'
    img = cv2.imread(image_path, 0)
    img = torch.tensor(img).to(device)
    img = img.unsqueeze(0).unsqueeze(0).double()/255

    import pdb;
    pdb.set_trace()
    coeffs = fb.build(img)