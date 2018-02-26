# tensor_estimation
Estimate diffusion tensors from dMRI data. 

## Dependencies
Nibabel 2.2.1 (IO)
Dipy 0.13.0 (IO)
Numpy 1.13.3

## Usage
Given dwi volumes (nifti format), a binary brainmask (nifti format), bvecs (tab delim txt), and bvals (tab delim txt), will compute tensors at each voxel in mask and output FA, MD, RD, and AD maps. 

python fit_tensors.py -dwi <filename.nii.gz> -mask <filename.nii.gz> -bvecs <filename.txt> -bvals <filename.txt> -out <prefix>