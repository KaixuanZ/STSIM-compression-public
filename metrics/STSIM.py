from __future__ import division
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from filterbank.VGG import VGG
from filterbank.Spyr_PyTorch import Spyr_PyTorch
from filterbank.SCFpyr_PyTorch import SCFpyr_PyTorch
from filterbank.DCT import DCT
from filterbank.sp3Filters import sp3Filters
from filterbank.sp0Filters import sp0Filters

class MyLinear(nn.Module):
	# parameters are between 0 and 1
	def __init__(self, input_size, output_size):
		super().__init__()
		self.W = nn.Parameter(torch.zeros(input_size, output_size))
		nn.init.xavier_uniform(self.W)

	def forward(self, x):
		return torch.mm(x, self.W**2)

class Metric:
	# implementation of STSIM global (no sliding window), as the global version has a better performance, and also easier to implement
	def __init__(self, filter, device, height=5, nbands=4, sub_sample=True, dim=82, blocksize=None, stepsize=None):
		params = vars()
		del params['self']
		self.__dict__.update(params)

		self.C = 1e-10
		if self.filter == 'SF':
			if self.nbands == 1:	# dim = 22
				filter = sp0Filters(self.device)
			elif self.nbands == 4:	# dim = 82
				filter = sp3Filters(self.device)
			self.fb = Spyr_PyTorch(filter, height=self.height, nbands=self.nbands, sub_sample = sub_sample, device = self.device)
		elif self.filter == 'SCF':
			self.fb = SCFpyr_PyTorch(height=self.height, nbands=self.nbands, sub_sample = sub_sample, device = self.device)
		elif self.filter == 'DCT':
			self.fb = DCT(device = self.device)
		elif self.filter == 'VGG':
			self.fb = VGG().to(self.device).double()

	def STSIM1(self, img1, img2, sub_sample=True):
		assert img1.shape == img2.shape
		assert len(img1.shape) == 4  # [N,C,H,W]
		assert img1.shape[1] == 1	# gray image

		pyrA = self.fb.getlist(self.fb.build(img1))
		pyrB = self.fb.getlist(self.fb.build(img2))

		stsim = map(self.pooling, pyrA, pyrB)

		return torch.mean(torch.stack(list(stsim)), dim=0).T # [BatchSize, FeatureSize]

	def STSIM2(self, img1, img2, sub_sample=True):
		assert img1.shape == img2.shape

		pyrA = self.fb.build(img1)
		pyrB = self.fb.build(img2)
		stsimg2 = list(map(self.pooling, self.fb.getlist(pyrA), self.fb.getlist(pyrB)))

		Nor = len(pyrA[1])

		# Accross scale, same orientation
		for scale in range(2, len(pyrA) - 1):
			for orient in range(Nor):
				img11 = pyrA[scale - 1][orient]
				img12 = pyrA[scale][orient]
				img11 = F.interpolate(img11, size=img12.shape[2:])

				img21 = pyrB[scale - 1][orient]
				img22 = pyrB[scale][orient]
				img21 = F.interpolate(img21, size=img22.shape[2:])

				stsimg2.append(self.compute_cross_term(img11, img12, img21, img22))

		# Accross orientation, same scale
		for scale in range(1, len(pyrA) - 1):
			for orient in range(Nor - 1):
				img11 = pyrA[scale][orient]
				img21 = pyrB[scale][orient]

				for orient2 in range(orient + 1, Nor):
					img13 = pyrA[scale][orient2]
					img23 = pyrB[scale][orient2]
					stsimg2.append(self.compute_cross_term(img11, img13, img21, img23))

		return torch.mean(torch.stack(stsimg2), dim=0).T # [BatchSize, FeatureSize]

	def pooling(self, img1, img2):
		# img1 = torch.abs(img1)
		# img2 = torch.abs(img2)
		tmp = self.compute_L_term(img1, img2) * self.compute_C_term(img1, img2) * self.compute_C01_term(img1,
																										img2) * self.compute_C10_term(
			img1, img2)
		return tmp ** 0.25

	def compute_L_term(self, img1, img2):
		# expectation over a small window
		mu1 = torch.mean(img1, dim=[1, 2, 3])
		mu2 = torch.mean(img2, dim=[1, 2, 3])

		Lmap = (2 * mu1 * mu2 + self.C) / (mu1 * mu1 + mu2 * mu2 + self.C)
		return Lmap

	def compute_C_term(self, img1, img2):
		mu1 = torch.mean(img1, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu2 = torch.mean(img2, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)

		sigma1_sq = torch.mean((img1 - mu1) ** 2, dim=[1, 2, 3])
		sigma1 = torch.sqrt(sigma1_sq)
		sigma2_sq = torch.mean((img2 - mu2) ** 2, dim=[1, 2, 3])
		sigma2 = torch.sqrt(sigma2_sq)

		Cmap = (2 * sigma1 * sigma2 + self.C) / (sigma1_sq + sigma2_sq + self.C)
		return Cmap

	def compute_C01_term(self, img1, img2):
		img11 = img1[..., :-1]
		img12 = img1[..., 1:]
		img21 = img2[..., :-1]
		img22 = img2[..., 1:]

		mu11 = torch.mean(img11, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu12 = torch.mean(img12, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu21 = torch.mean(img21, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu22 = torch.mean(img22, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)

		sigma11_sq = torch.mean((img11 - mu11) ** 2, dim=[1, 2, 3])
		sigma12_sq = torch.mean((img12 - mu12) ** 2, dim=[1, 2, 3])
		sigma21_sq = torch.mean((img21 - mu21) ** 2, dim=[1, 2, 3])
		sigma22_sq = torch.mean((img22 - mu22) ** 2, dim=[1, 2, 3])

		sigma1_cross = torch.mean((img11 - mu11) * (img12 - mu12), dim=[1, 2, 3])
		sigma2_cross = torch.mean((img21 - mu21) * (img22 - mu22), dim=[1, 2, 3])

		rho1 = (sigma1_cross + self.C) / (torch.sqrt(sigma11_sq * sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C) / (torch.sqrt(sigma21_sq * sigma22_sq) + self.C)

		C01map = 1 - 0.5 * torch.abs(rho1 - rho2)

		return C01map

	def compute_C10_term(self, img1, img2):
		img11 = img1[:, :, :-1, :]
		img12 = img1[:, :, 1:, :]
		img21 = img2[:, :, :-1, :]
		img22 = img2[:, :, 1:, :]

		mu11 = torch.mean(img11, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu12 = torch.mean(img12, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu21 = torch.mean(img21, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu22 = torch.mean(img22, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)

		sigma11_sq = torch.mean((img11 - mu11) ** 2, dim=[1, 2, 3])
		sigma12_sq = torch.mean((img12 - mu12) ** 2, dim=[1, 2, 3])
		sigma21_sq = torch.mean((img21 - mu21) ** 2, dim=[1, 2, 3])
		sigma22_sq = torch.mean((img22 - mu22) ** 2, dim=[1, 2, 3])

		sigma1_cross = torch.mean((img11 - mu11) * (img12 - mu12), dim=[1, 2, 3])
		sigma2_cross = torch.mean((img21 - mu21) * (img22 - mu22), dim=[1, 2, 3])

		rho1 = (sigma1_cross + self.C) / (torch.sqrt(sigma11_sq) * torch.sqrt(sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C) / (torch.sqrt(sigma21_sq) * torch.sqrt(sigma22_sq) + self.C)

		C10map = 1 - 0.5 * torch.abs(rho1 - rho2)

		return C10map

	def compute_cross_term(self, img11, img12, img21, img22):
		mu11 = torch.mean(img11, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu12 = torch.mean(img12, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu21 = torch.mean(img21, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
		mu22 = torch.mean(img22, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)

		sigma11_sq = torch.mean((img11 - mu11) ** 2, dim=[1, 2, 3])
		sigma12_sq = torch.mean((img12 - mu12) ** 2, dim=[1, 2, 3])
		sigma21_sq = torch.mean((img21 - mu21) ** 2, dim=[1, 2, 3])
		sigma22_sq = torch.mean((img22 - mu22) ** 2, dim=[1, 2, 3])
		sigma1_cross = torch.mean((img11 - mu11) * (img12 - mu12), dim=[1, 2, 3])
		sigma2_cross = torch.mean((img21 - mu21) * (img22 - mu22), dim=[1, 2, 3])

		rho1 = (sigma1_cross + self.C) / (torch.sqrt(sigma11_sq * sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C) / (torch.sqrt(sigma21_sq * sigma22_sq) + self.C)

		Crossmap = 1 - 0.5 * torch.abs(rho1 - rho2)
		return Crossmap

	def _STSIM_with_mask(self, img, mask):
		'''
		:param img: [N,C=1,H,W]
		:return: [N, feature dim] STSIM-features
		return STSIM features given a mask
		'''
		coeffs = self.fb.build(img)
		if self.filter == 'SCF':  # magnitude of coeff
			for i in range(1, 4):
				for j in range(0, 4):
					coeffs[i][j] = torch.view_as_complex(coeffs[i][j]).abs()
			for i in [0, -1]:
				coeffs[i] = coeffs[i].abs()
		# elif self.filter == 'SF':	# magnitude of coeff
		#	for i in range(1,4):
		#		for j in range(0,4):
		#			coeffs[i][j] = coeffs[i][j].abs()
		#	for i in [0, -1]:
		#		coeffs[i] = coeffs[i].abs()

		f = []
		# single subband statistics
		for c in self.fb.getlist(coeffs):
			# mask the subband
			k = mask.shape[-1]//c.shape[-1]
			mask_c = mask[:,:,::k,::k]
			# stats with mask
			mean = torch.sum(c*mask_c, dim=[1, 2, 3])/mask_c.sum()
			f.append(mean)
			c1 = c - mean.reshape([-1, 1, 1, 1])
			var = torch.sum((c1*mask_c)**2, dim=[1, 2, 3])/(mask_c.sum()-1+self.C)
			f.append(var)
			f.append(torch.sum(c1[:, :, :-1, :] * c1[:, :, 1:, :] * mask_c[:,:,:-1,:], dim=[1, 2, 3]) / ((mask_c.sum()-1) * var + self.C))
			f.append(torch.sum(c1[:, :, :, :-1] * c1[:, :, :, 1:] * mask_c[:,:,:,:-1], dim=[1, 2, 3]) / ((mask_c.sum()-1) * var + self.C))

		# correlation statistics
		# across orientations
		for orients in coeffs[1:-1]:
			for (c1, c2) in list(itertools.combinations(orients, 2)):
				k = mask.shape[-1] // c1.shape[-1]
				mask_c = mask[:, :, ::k, ::k]

				mean_c1 = torch.sum(c1*mask_c, dim=[1, 2, 3]) / mask_c.sum()
				mean_c2 = torch.sum(c2*mask_c, dim=[1, 2, 3]) / mask_c.sum()
				c1 = c1 - mean_c1.reshape([-1,1,1,1])
				c2 = c2 - mean_c2.reshape([-1,1,1,1])
				var_c1 = torch.sum((c1 * mask_c) ** 2, dim=[1, 2, 3]) / (mask_c.sum() - 1 + self.C)
				var_c2 = torch.sum((c2 * mask_c) ** 2, dim=[1, 2, 3]) / (mask_c.sum() - 1 + self.C)
				denom = torch.sqrt(var_c1 * var_c2) + self.C
				nom = torch.sum(c1*c2*mask_c, dim=[1, 2, 3])/mask_c.sum()
				f.append(nom / denom)

		for orient in range(len(coeffs[1])):
			for height in range(len(coeffs) - 3):
				c1 = coeffs[height + 1][orient]
				c2 = coeffs[height + 2][orient]
				c1 = F.interpolate(c1, size=c2.shape[2:])
				k = mask.shape[-1] // c2.shape[-1]
				mask_c = mask[:, :, ::k, ::k]

				mean_c1 = torch.sum(c1 * mask_c, dim=[1, 2, 3]) / mask_c.sum()
				c1 = c1 - mean_c1.reshape([-1,1,1,1])
				mean_c2 = torch.sum(c2 * mask_c, dim=[1, 2, 3]) / mask_c.sum()
				c2 = c2 - mean_c2.reshape([-1,1,1,1])

				var_c1 = torch.sum((c1 * mask_c) ** 2, dim=[1, 2, 3]) / (mask_c.sum() - 1 + self.C)
				var_c2 = torch.sum((c2 * mask_c) ** 2, dim=[1, 2, 3]) / (mask_c.sum() - 1 + self.C)
				denom = torch.sqrt(var_c1 * var_c2) + self.C
				nom = torch.sum(c1 * c2 * mask_c, dim=[1, 2, 3]) / mask_c.sum()
				f.append(nom / denom)

		return torch.stack(f).T  # [BatchSize, FeatureSize]

	def _STSIM(self, imgs, reshape=False):
		'''
		:param imgs: [N,C=1,H,W]
		:return: [N, H//bs, W//bs, feature dim] STSIM-features
		'''
		coeffs = self.fb.build(imgs)
		if self.filter == 'SCF':	# magnitude of coeff
			for i in range(1,self.height-1):
				for j in range(0,self.nbands):
					coeffs[i][j] = torch.view_as_complex(coeffs[i][j]).abs()
			for i in [0, -1]:
				coeffs[i] = coeffs[i].abs()

		I = (coeffs[0].shape[2] - self.blocksize)//self.stepsize + 1
		J = (coeffs[0].shape[3] - self.blocksize)//self.stepsize + 1	# block counter

		# compute block size in current subband
		def curr_blocksize(coeff_ref, coeff_cur):
			k = coeff_ref.shape[-1]//coeff_cur.shape[-1]
			return self.blocksize//k

		tmp_I = []
		for i in range(I):
			tmp_J = []
			for j in range(J):
				f = []

				# single subband statistics
				for coeff in self.fb.getlist(coeffs):
					bs = curr_blocksize(coeffs[0], coeff)
					ss = bs//self.blocksize*self.stepsize
					c = coeff[:, :, i*ss:i*ss+bs, j*ss:j*ss+bs]

					mean = torch.mean(c, dim=[1, 2, 3])
					var = torch.var(c, dim=[1, 2, 3])

					f.append(mean)
					f.append(var)

					c = c - mean.reshape([-1, 1, 1, 1])
					f.append(torch.mean(c[:, :, :-1, :] * c[:, :, 1:, :], dim=[1, 2, 3]) / (var + self.C))
					f.append(torch.mean(c[:, :, :, :-1] * c[:, :, :, 1:], dim=[1, 2, 3]) / (var + self.C))

				# correlation statistics
				# across orientations
				for orients in coeffs[1:-1]:
					for (coeff1, coeff2) in list(itertools.combinations(orients, 2)):
						bs = curr_blocksize(coeffs[0], coeff1)
						ss = bs // self.blocksize * self.stepsize
						c1 = coeff1[:, :, i*ss:i*ss+bs, j*ss:j*ss+bs]
						c2 = coeff2[:, :, i*ss:i*ss+bs, j*ss:j*ss+bs]

						c1 = torch.abs(c1)
						c1 = c1 - torch.mean(c1, dim=[1, 2, 3]).reshape([-1, 1, 1, 1])
						c2 = torch.abs(c2)
						c2 = c2 - torch.mean(c2, dim=[1, 2, 3]).reshape([-1, 1, 1, 1])
						denom = torch.sqrt(torch.var(c1, dim=[1, 2, 3]) * torch.var(c2, dim=[1, 2, 3]))
						f.append(torch.mean(c1 * c2, dim=[1, 2, 3]) / (denom + self.C))

				# across scales
				for orient in range(len(coeffs[1])):
					for height in range(len(coeffs) - 3):
						coeff1 = torch.abs(coeffs[height + 1][orient])
						bs = curr_blocksize(coeffs[0], coeff1)
						ss = bs // self.blocksize * self.stepsize
						c1 = coeff1[:, :, i*ss:i*ss+bs, j*ss:j*ss+bs]
						c1 = c1 - torch.mean(c1, dim=[1, 2, 3]).reshape([-1, 1, 1, 1])

						coeff2 = torch.abs(coeffs[height + 2][orient])
						bs = curr_blocksize(coeffs[0], coeff2)
						ss = bs // self.blocksize * self.stepsize
						c2 = coeff2[:, :, i*ss:i*ss+bs, j*ss:j*ss+bs]
						c2 = c2 - torch.mean(c2, dim=[1, 2, 3]).reshape([-1, 1, 1, 1])
						c1 = F.interpolate(c1, size=c2.shape[2:])
						denom = torch.sqrt(torch.var(c1, dim=[1, 2, 3]) * torch.var(c2, dim=[1, 2, 3]))
						f.append(torch.mean(c1 * c2, dim=[1, 2, 3]) / (denom + self.C))
				tmp_J.append(torch.stack(f, dim=1))  # [BatchSize, FeatureSize]
			tmp_I.append(torch.stack(tmp_J, dim=1))
		res = torch.stack(tmp_I, dim=1) #[N, H//bs, W//bs, feature dim]

		assert res.shape[-1] == self.dim
		if reshape:
			return res.reshape(-1, self.dim)
		else:
			return res

	def STSIM(self, img, mask=None):
		'''
		:param img: [N,C=1,H,W]
		:return: [N, feature dim] STSIM-features
		'''
		if self.blocksize is not None:
			return self._STSIM(img)
		if mask is not None:
			return self._STSIM_with_mask(img, mask)
		if self.filter=='VGG':
			img = F.interpolate(img, size=256).to(self.device)

		coeffs = self.fb.build(img)
		if self.filter == 'SCF':	# magnitude of coeff
			for i in range(1,4):
				for j in range(0,4):
					coeffs[i][j] = torch.view_as_complex(coeffs[i][j]).abs()
			for i in [0, -1]:
				coeffs[i] = coeffs[i].abs()
		#elif self.filter == 'SF':	# magnitude of coeff
		#	for i in range(1,4):
		#		for j in range(0,4):
		#			coeffs[i][j] = coeffs[i][j].abs()
		#	for i in [0, -1]:
		#		coeffs[i] = coeffs[i].abs()

		f = []
		# single subband statistics
		for c in self.fb.getlist(coeffs):
			mean = torch.mean(c, dim = [1,2,3])
			var = torch.var(c, dim = [1,2,3])
			f.append(mean)
			f.append(var)

			c = c - mean.reshape([-1,1,1,1])
			f.append(torch.mean(c[:, :, :-1, :] * c[:, :, 1:, :], dim=[1, 2, 3]) / (var + self.C))
			f.append(torch.mean(c[:, :, :, :-1] * c[:, :, :, 1:], dim=[1, 2, 3]) / (var + self.C))
		if self.filter == 'VGG':
			return torch.stack(f).T
		# correlation statistics
		# across orientations
		for orients in coeffs[1:-1]:
			for (c1, c2) in list(itertools.combinations(orients, 2)):
				c1 = c1 - torch.mean(c1, dim = [1,2,3]).reshape([-1,1,1,1])
				c2 = c2 - torch.mean(c2, dim = [1,2,3]).reshape([-1,1,1,1])
				denom = torch.sqrt(torch.var(c1, dim = [1,2,3]) * torch.var(c2, dim = [1,2,3]))
				f.append(torch.mean(c1*c2, dim = [1,2,3])/(denom + self.C))

		for orient in range(len(coeffs[1])):
			for height in range(len(coeffs) - 3):
				c1 = coeffs[height + 1][orient]
				c2 = coeffs[height + 2][orient]
				c1 = F.interpolate(c1, size=c2.shape[2:])

				c1 = c1 - torch.mean(c1, dim=[1, 2, 3]).reshape([-1, 1, 1, 1])
				c2 = c2 - torch.mean(c2, dim=[1, 2, 3]).reshape([-1,1,1,1])
				denom = torch.sqrt(torch.var(c1, dim = [1,2,3]) * torch.var(c2, dim = [1,2,3]))
				f.append(torch.mean(c1*c2, dim = [1,2,3])/(denom + self.C))
		return torch.stack(f).T # [BatchSize, FeatureSize]

	def STSIM_M(self, X1, X2=None, weight=None):
		if weight is not None:
			if len(X1.shape) == 4:
				# the input are raw images, extract STSIM-M features
				with torch.no_grad():
					X1 = self.STSIM(X1)  # [N, dim of feature]
					X2 = self.STSIM(X2)
			pred = (X1-X2)/weight	#[N, dim of feature]
			pred = torch.sqrt(torch.sum(pred**2,1))
			return pred
		else:
			if len(X1.shape) == 4:
				# the input are raw images, extract STSIM-M features
				with torch.no_grad():
					X1 = self.STSIM(X1)  # [N, dim of feature]
			weight = X1.std(0) + 1e-20	#[dim of feature]
			return weight

	def STSIM_I(self, X1, X2=None, mask=None, weight=None):
		if weight is not None:
			return self.STSIM_M(X1,X2,weight)
		else:
			if len(X1.shape) == 4:
				# the input are raw images, extract STSIM-M features
				with torch.no_grad():
					X1 = self.STSIM(X1)  # [N, dim of feature]
			var = torch.zeros(X1.shape[1], device=self.device)
			for i in set(mask.detach().cpu().numpy()):
				X1_i = X1[mask==i]
				X1_i = X1_i - X1_i.mean(0)	# substract intra-class mean [N, dim of feature]
				var += (X1_i**2).sum(0)		# square sum	for all intra-class sample [dim of feature]
			return torch.sqrt(var/X1.shape[0]) + 1e-20

class STSIM_M(torch.nn.Module):
	def __init__(self, dim, mode=0, filter=None, device=None):
		'''
		Args:
			mode: regression, STSIM-M
			weights_path:
		'''
		super(STSIM_M, self).__init__()

		self.device = torch.device('cpu') if device is None else device
		self.mode = mode
		self.filter = filter
		if self.mode == 0:  	# factorization
			self.linear = nn.Linear(dim[0], dim[1])
			# self.linear = nn.Linear(dim[0], dim[1], bias=False)
		elif self.mode == 1:  	# 3-layer neural net
			self.hidden = nn.Linear(dim[0], dim[0])
			self.predict = nn.Linear(dim[0], 1)
		elif self.mode == 2:  	# regression
			self.linear = nn.Linear(dim[0], 1)
		elif self.mode == 3:	# diagonal Mahalanobis
			#self.linear = nn.Linear(dim[0], 1)
			self.linear = MyLinear(dim[0], 1)

	def forward(self, X1, X2):
		'''
		Args:
			X1:
			X2:
		Returns:
		'''
		if len(X1.shape) == 4:
			# the input are raw images, extract STSIM-M features
			m = Metric(self.filter, device=self.device)
			with torch.no_grad():
				X1 = m.STSIM(X1)
				X2 = m.STSIM(X2)
		if self.mode == 0:  # STSIM_Mf
			pred = self.linear(torch.abs(X1 - X2))  # [N, dim]
			pred = torch.bmm(pred.unsqueeze(1), pred.unsqueeze(-1)).squeeze(-1)  # inner-prod
			return torch.sqrt(pred) - torch.abs(torch.sum(self.linear.bias))  # [N, 1]
			# return torch.sqrt(pred)  # [N, 1]
			# return torch.sqrt(pred)# [N, 1]
		elif self.mode == 1:  # 3-layer neural net STSIM-NN
			pred = F.relu(self.hidden(torch.abs(X1 - X2)))
			pred = torch.sigmoid(self.predict(pred))
			return pred
		elif self.mode == 2:  # regression
			pred = self.linear(torch.abs(X1 - X2))  # [N, 1]
			return torch.sigmoid(pred)
		elif self.mode == 3:  # STSIM (diagonal) data driven STSIM-Md
			pred = self.linear(torch.abs(X1 - X2)**2)  # [N, 1]
			return torch.sqrt(pred) #- torch.abs(torch.sum(self.linear.bias))
		elif self.mode == 4:
			pred = self.linear(torch.abs(X1 - X2))  # [N, dim]
			pred = torch.bmm(pred.unsqueeze(1), pred.unsqueeze(-1)).squeeze(-1)  # inner-prod
			return self.scale(torch.sqrt(pred))  # [N, 1]

if __name__ == '__main__':
	def test1():
		import cv2
		import numpy as np
		img_o = cv2.imread('../data/original.png', 0).astype(float)
		img_den = cv2.imread('../data/denoised.png', 0).astype(float)

		fg = img_o - img_den

		# mask of flat region
		edges = cv2.Canny(img_den.astype(np.uint8), 50, 100)
		kernel = np.ones((5, 5), np.uint8)
		mask = cv2.dilate(edges, kernel, iterations=1)
		mask = 1 - mask/255

		mask = torch.tensor(mask).double()
		fg = torch.tensor(fg).double()
		fg = fg.unsqueeze(0).unsqueeze(0)
		mask = mask.unsqueeze(0).unsqueeze(0)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		m = Metric('SF', device)

		fg = fg.double().to(device)

		feature_img1 = m.STSIM(fg.repeat(2,1,1,1), mask)

		import pdb;pdb.set_trace()

	def test2():
		from PIL import Image, ImageOps
		import torchvision.transforms as transforms

		img_path = '../../noise_analysis/noise_decoder_f300.png'
		img = ImageOps.grayscale(Image.open(img_path))
		img = transforms.ToTensor()(img)
		#img = img.unsqueeze(0)  # batchsize = 1
		img = torch.stack([img,img],dim=0)  # batchsize = 2

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		m1 = Metric('SF', height=5, nbands=4, device=device, dim=82, blocksize=64, stepsize=32)
		res1 = m1.STSIM(img.double().to(device))

		m2 = Metric('SCF', height=5, nbands=1, device=device, dim=22, blocksize=128, stepsize=64)
		res2 = m2.STSIM(img.double().to(device))

		m3 = Metric('SCF', height=5, nbands=1, device=device, dim=22, blocksize=128, stepsize=128)
		res3 = m3.STSIM(img.double().to(device))
		import pdb;
		pdb.set_trace()
	test2()