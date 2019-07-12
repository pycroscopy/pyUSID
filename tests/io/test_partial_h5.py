"""
This script creates a partial h5py file then tests the process class with it.
Created on: Jul 12, 2019
Author: Emily Costa
"""

from data_utils import make_sparse_sampling_file
import pyUSID as usid
from pyUSID.io import dtype_utils, hdf_utils
import h5py


# Creates incomplete h5py dataset object in current path
h5_meas_grp = make_sparse_sampling_file()
h5_path = 'sparse_sampling.h5'
h5_f = h5py.File(h5_path, mode='r+')
hdf_utils.print_tree(h5_f)
#hdf_utils.simple.check_if_main(h5_main, verbose=True)
#dtype_utils.check_dtype(h5_main)

