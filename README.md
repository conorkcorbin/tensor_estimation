# tensor_estimation
Estimate diffusion tensors from dMRI data. 

## Dependencies
Nibabel 2.2.1 (IO)
Dipy 0.13.0 (IO)
Numpy 1.13.3

## Usage
Given dwi volumes, a binary brainmask, bvecs, and bvals, will compute tensors at each voxel in mask and output FA, MD, RD, and AD maps. 

python fit_tensors.py -dwi <filename.nii.gz> -mask <filename.nii.gz> -bvecs <filename.txt> -bvals <filename.txt> -out <prefix>