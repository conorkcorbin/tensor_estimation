import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs

class TensorEstimator:

    def __init__(self, f_dwi, f_mask, f_bvecs, f_bvals, f_out):
        """ Initialize tensor estimation object
        
        Parameters
        ----------
        f_dwi: string 
            file path to diffusion image
        f_mask : string 
            file path to mask
        f_bvecs : string
            file path to bvecs - unit vectors plz
        f_bvals : string
            file path to bvals
        f_out : string
            out prefix


        """ 
        dwi_img = nib.load(f_dwi)
        mask_img = nib.load(f_mask)
        dwi_data = dwi_img.get_data()

        bvals, bvecs = read_bvals_bvecs(f_bvals, f_bvecs)
        bzero_indices = np.argwhere(bvals==0).flatten()
        dwi_indices = [i for i in range(len(bvals))
                            if i not in bzero_indices]

        self.bzeros = np.delete(dwi_data, dwi_indices, axis=3)
        self.dwi = np.delete(dwi_data, bzero_indices, axis=3)
        self.bval = np.delete(bvals, bzero_indices)[0]
        self.bvecs = np.delete(bvecs, bzero_indices, axis=0)
        self.mask = mask_img.get_data()
        shape = self.mask.shape
        self.mask_indices = [(i, j, k) for k in range(shape[2])
                                       for j in range(shape[1])
                                       for i in range(shape[0])
                                       if (self.mask[i, j, k] == 1 and 
                                       not np.any(self.dwi[i, j, k, :] == 0))]  
        self.affine = mask_img.get_affine()
        self.f_out = f_out

    def fit(self):
        """ 
        Fit diffusion tensors to every voxel in mask 
        Least Squares Estimate

        Calculate FA, MD, RD, AD from tensors and save to nifti files

        """
        tensors = [self._fit_voxel(self.dwi[i[0], i[1], i[2], :],
                              self.bzeros[i[0], i[1], i[2], :],
                              self.bval,
                              self.bvecs)
                              for i in self.mask_indices]
        print("Tensors Estimated")

        mean_diffusivity = [self._mean_diffusivity(t) for t in tensors]
        print("MD calculated")
        axial_diffusivity = [self._axial_diffusivity(t) for t in tensors]
        print("AD calculated")
        radial_diffusivity = [self._radial_diffusivity(t) for t in tensors]
        print("RD calculated")
        fractional_anisotropy = [self._fractional_anisotropy(t) for t in tensors]
        print("FA calculated")

        self._save_nifti(mean_diffusivity, 'MD')
        self._save_nifti(axial_diffusivity, 'AD')
        self._save_nifti(radial_diffusivity, 'RD')
        self._save_nifti(fractional_anisotropy, 'FA')
        print("Saved Niftis")

    def _fit_voxel(self, Sk, So, bval, bvecs):
        """ 
        Fits a diffusion tensor at the voxel with dwi signals
        Sk and bzero signal So.

        Parameters
        ----------
        Sk : list
            of diffusion weighted intensity values at voxel
        So : list
             of intensities at voxel for each bzero
        bval : float 
            bval of the dwi
        bvecs : np array shape = (N x 3)
            unit vector describing direction of each dwi sig

        Returns
        -------
        tensor : np array shape = (3 x 3)
            estimated diffusion tensor at this voxel

        """
        So = np.average(So)
        Y = np.divide(np.log(np.divide(So, Sk)), bval) # shape = (N x 1)
        X = np.array([np.array([g[0]**2, g[1]**2,
                                g[2]**2,2*g[0]*g[1],
                                2*g[0]*g[2], 2*g[1]*g[2]])
                                for g in bvecs]) # shape = (N x 6)
        W = np.linalg.lstsq(X, Y) # shape = (1 x 6)
        W = W[0] # just solutions (not residuals)
        tensor = np.array([[W[0], W[3], W[4]],
                           [W[3], W[1], W[5]],
                           [W[4], W[5], W[2]]])
        return tensor

    def _mean_diffusivity(self, tensor):
        """ Returns mean diffusivity from tensor """

        evals, evecs = np.linalg.eig(tensor)
        # self.evals.append(np.sort(evals))
        return np.average(evals)

    def _fractional_anisotropy(self, tensor):
        evals, evecs = np.linalg.eig(tensor)

        mean_eval = np.average(evals)
        numerator = np.sqrt((evals[0]-mean_eval)**2 +
                            (evals[1]-mean_eval)**2 +
                            (evals[2]-mean_eval)**2)
        denominator = np.sqrt(evals[0]**2 +
                              evals[1]**2 +
                              evals[2]**2)
        return np.sqrt(3.0/2.0) * (numerator/denominator)

    def _axial_diffusivity(self, tensor):
        evals, evecs = np.linalg.eig(tensor)
        return np.max(evals)

    def _radial_diffusivity(self, tensor):
        evals, evecs = np.linalg.eig(tensor)
        evals = np.sort(evals)
        return (evals[0] + evals[1])/2

    def _save_nifti(self, measure, suffix):
        """ Saves a nifti of the computed measure """ 
        shape = self.mask.shape
        image_data = np.zeros(shape)
        for i, ind in enumerate(self.mask_indices):
            image_data[ind[0], ind[1], ind[2]] = measure[i]
        image = nib.Nifti1Image(image_data, self.affine)
        nib.save(image, ''.join([self.f_out, '_', suffix, '.nii.gz']))



