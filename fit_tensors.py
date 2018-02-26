import argparse
import pandas as pd
import numpy as np
import nibabel as nib
import os
import sys

from TensorEstimator import TensorEstimator
# sys.path.append('/Volumes/four_d/ccorbin/Frontiers/ADNI3/002_S_0413_1/dwi/')
# # /Volumes/four_d/ccorbin/Frontiers/ADNI3/002_S_0413_1/dwi

# # Read Arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-dwi', '--dwi', required = True)
# parser.add_argument('-mask', '--mask', required = True)
# parser.add_argument('-bvecs', '--bvecs', required = True)
# parser.add_argument('-bvals', '--bvals', required = True)
# args = parser.parse_args()

# os.system('ls /Volumes/four_d/ccorbin/Frontiers/ADNI3/002_S_0413_1/dwi/')

# # Load Data
# dwi_img = nib.load(args.dwi)
# mask_img = nib.load(args.mask)
# dwi_data = dwi_img.get_data()
# mask_data = mask_img.get_data()

# bvecs = pd.read_csv(args.bvecs, sep='\t', header=False)
# bvals = pd.read_csv(args.bvecs, sep='\t', header=False)
# bvecs = bvecs.values
# bvals = bvals.values

# print(bvecs)
# print(bvals)

# Read Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-data', '--data', required = True)
args = parser.parse_args()

# Read / Parse data
data = pd.read_csv(args.data)
bvecs = data[['Gx', 'Gy', 'Gz']].values
dwi = data['Intensity'].values
bvals = data['bvals'].values

index = np.argwhere(bvals==0)[0]
bzero = dwi[index]
dwi = np.delete(dwi, index)
bvecs = np.delete(bvecs, index, axis=0)
bvals = np.delete(bvals, index)
bval = bvals[0]

te = TensorEstimator()
tensor = te._fit_voxel(dwi, bzero, bval, bvecs)
