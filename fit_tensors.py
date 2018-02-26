import argparse
import pandas as pd
import numpy as np
import nibabel as nib
import os
import sys
from dipy.io import read_bvals_bvecs

from TensorEstimator import TensorEstimator

# Read Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dwi', '--dwi', required = True)
parser.add_argument('-mask', '--mask', required = True)
parser.add_argument('-bvecs', '--bvecs', required = True)
parser.add_argument('-bvals', '--bvals', required = True)
parser.add_argument('-out', '--out', required = True)
args = parser.parse_args()

te = TensorEstimator(args.dwi, args.mask, args.bvecs, args.bvals, args.out)
te.fit()

