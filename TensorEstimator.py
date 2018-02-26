import numpy as np

class TensorEstimator:

	def __init__(self):
		""" Initialize tensor estimation object
		
		Parameters
		----------
		dwi : np array - shape = [N, X, Y ,Z]
			diffusion signal from from each of N directions
		mask : binary mask (gives indices where we will fit tensors)
		bvecs : np array - shape = [N, 3]
		bvals : np array - shape = [N x 1]

		""" 
		# self._dwi = dwi
		# self._mask = mask
		# self._bvecs = bvecs[1:]
		# self._bvals = bvals
		# self._bval = self._bvals[:-1] # assume last bval will be non-zero

		# # for now just assume first volume is b0 and only single shell
		# # index_bzero = np.argwhere(self._bvals==0)[0][0]
		# # self._bzero = self._dwi[index_bzero]
		# self._bzero = self._dwi[0, :, :, :]
		# self._dwi = self._dwi[1:, :, :, :]


	def fit(self):
		""" fit diffusion tensors to every voxel in mask 

		Uses least squares to fit tensor values from equation 
		Y = WX 
		Where Y = ln(So/Sk)/bval
		W (the tensor vals we need to fit) : Dxx Dyy Dzz Dxy Dyz Dxz 
			we know the tensor is a SPD symmetric matrix 
		X : bvec coefficients
		
		# a 4D array of Y's 
		"""

		# Y_vol = np.divide(np.log(np.divide(self._bzero, self._dwi)), self._bval) 


	def _fit_voxel(self, Sk, So, bval, bvecs):
		""" 
		Fits a diffusion tensor at the voxel with dwi signals Sk and 
		bzero signal So.

		Parameters
		----------
		Sk : list
			of diffusion weighted intensity values at voxel
		So : float
			intensity at voxel of bzero
		bval : float 
			bval of the dwi
		bvecs : np array shape = (N x 3)
			unit vector describing direction of each dwi sig

		Returns
		-------
		tensor : np array shape = (3 x 3)
			estimated diffusion tensor at this voxel

		"""
		# Y = np.divide(np.log(np.divide(self._bzero, dwi)), self._bval)
		# X = [[g[0] ** 2, g[1] ** 2, g[2] ** 2, g[0]*g[1], g[1]*g[3], g[2]*g[3]]
		# 		for g in self._bvecs]
		# tensor = np.linalg.lstsq(X, Y)

		Y = np.divide(np.log(np.divide(So, Sk)), bval) # shape = (N x 1)
		print("log(so/sk) / b")
		print(Y.shape)
		print(Y)
		X = np.array([np.array([g[0]**2, g[1]**2, g[2]**2, 2*g[0]*g[1], 2*g[0]*g[2], 2*g[1]*g[2]])
				for g in bvecs]) # shape = (N x 6)
		print("Coefficients from bvecs")
		print(X.shape)
		print(X)
		W = np.linalg.lstsq(X, Y) # shape = (N x 6)
		W = W[0]
		print("least squares solution")
		print(W.shape)
		print(W)
		tensor = np.array([[W[0], W[3], W[4]],
						   [W[3], W[1], W[5]],
						   [W[4], W[5], W[2]]])
		print("Estimated Tensor")
		print(tensor)


		# Cramer's rule
		W = np.linalg.solve(X, Y)
		evals, evecs = np.linalg.eig(tensor)
		print("Cramer's solution")
		print(W)
		print(evals)
		print(evecs)
		return tensor





