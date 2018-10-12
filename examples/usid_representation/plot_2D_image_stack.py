"""
================================================================================
04. 2D Image Stack
================================================================================

**Suhas Somnath**

10/12/2018

**This document is under construction.**

Please consider downloading this document as a Jupyter notebook using the
button at the bottom of this document.
"""
from __future__ import print_function, division, unicode_literals
import subprocess
import sys
import os
from warnings import warn
import h5py
import numpy as np
import matplotlib.pyplot as plt


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    # This package is not part of anaconda and may need to be installed.
    import wget
except ImportError:
    warn('wget not found.  Will install with pip.')
    import pip
    install('wget')
    import wget

# Finally import pyUSID.
try:
    import pyUSID as usid
except ImportError:
    warn('pyUSID not found.  Will install with pip.')
    import pip
    install('pyUSID')
    import pyUSID as usid

########################################################################################################################
# Download the dataset
# ---------------------
h5_path = 'temp.h5'
url = 'https://raw.githubusercontent.com/pycroscopy/pyUSID/master/data/SingFreqPFM_0003.h5'
if os.path.exists(h5_path):
    os.remove(h5_path)
_ = wget.download(url, h5_path, bar=None)

########################################################################################################################
# Open the file
# -------------
# Look at the contents of the file
h5_file = h5py.File(h5_path, mode='r')
usid.hdf_utils.print_tree(h5_file)

########################################################################################################################
# Notice that there are multiple channels with a single dataset named ``Raw_Data`` in them. From the `1D Spectrum`
# example, we now know that these ``Raw_Data`` datasets are actually ``Main`` datasets and they share the same
# set of ancillary datasets that are under ``Measurement_000`` group. This is common for Scanning Probe Microscopy
# scans where information from multiple sensors are recorded **simultaneously** during the scan.
#
# Visualize the contents in each of these channels
# ------------------------------------------------
usid.plot_utils.use_nice_plot_params()
for main_dset in usid.hdf_utils.get_all_main(h5_file):
    main_dset.visualize()

########################################################################################################################
# Access the ``Main`` Dataset containing the image of interest
# ------------------------------------------------------------
h5_main = usid.hdf_utils.get_all_main(h5_file)[-1]
print(h5_main)

########################################################################################################################
# Look at the Position Indices dataset linked to the Main dataset
# ---------------------------------------------------------------
print(h5_main.h5_pos_inds)

########################################################################################################################
# Look at the Position Values dataset linked to the Main dataset
# ---------------------------------------------------------------
print(h5_main.h5_pos_vals)

########################################################################################################################
# See the attributes within the Position Indices Dataset
# ------------------------------------------------------
for key, val in usid.hdf_utils.get_attributes(h5_main.h5_pos_inds).items():
    print('{} : {}'.format(key, val))


########################################################################################################################
# Visualize the contents of the Position Indices Dataset
# ------------------------------------------------------
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
axes[0].plot(h5_main.h5_pos_inds[()])
axes[0].set_title('Full dataset')
axes[1].set_title('First 1024 rows only')
axes[1].plot(h5_main.h5_pos_inds[:1024])
for axis in axes.flat:
    axis.set_xlabel('Row in Position Indices')
    axis.set_ylabel('Position Indices')
    axis.legend(['X', 'Y'])
fig.suptitle('Position Indices dataset')
fig.tight_layout()

########################################################################################################################
# Visualize the contents of the Position Values Dataset
# ------------------------------------------------------
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
axes[0].plot(h5_main.h5_pos_vals[()])
axes[0].set_title('Full dataset')
axes[1].set_title('First 1024 rows only')
axes[1].plot(h5_main.h5_pos_vals[:1024])
for axis in axes.flat:
    axis.set_xlabel('Row in Position Values')
    axis.set_ylabel('Position Values')
    axis.legend(['X', 'Y'])
fig.suptitle('Position Values dataset')
fig.tight_layout()

########################################################################################################################
# Look at the Spectroscopic Indices dataset linked to the Main dataset
# --------------------------------------------------------------------
# Given that this is a 2D image where data was not acquired as a function of some independent parameter at each
# location, the spectroscopic datasets will contain bare minimum information
print(h5_main.h5_spec_inds)

########################################################################################################################
# Look at the Spectroscopic Values dataset linked to the Main dataset
# --------------------------------------------------------------------
print(h5_main.h5_spec_vals)


########################################################################################################################
# See the attributes within the Spectroscopic Indices Dataset
# ------------------------------------------------------------
for key, val in usid.hdf_utils.get_attributes(h5_main.h5_spec_inds).items():
    print('{} : {}'.format(key, val))

########################################################################################################################
# See the contents within the Spectroscopic Datasets
# --------------------------------------------------
print(h5_main.h5_spec_inds[()], h5_main.h5_spec_vals[()])


########################################################################################################################
# Attempting to visualize the first few rows of the image manually
# ----------------------------------------------------------------
print(h5_main.pos_dim_labels)
print(h5_main.pos_dim_sizes)


########################################################################################################################
rows_to_read = 50
num_cols = h5_main.pos_dim_sizes[1]
first_few_rows_1D = h5_main[: rows_to_read * num_cols, :]
print(first_few_rows_1D.shape)


########################################################################################################################
first_few_rows_2D = np.reshape(first_few_rows_1D, (rows_to_read, num_cols))
print(first_few_rows_2D.shape)


########################################################################################################################
fig, axis = plt.subplots()
axis.imshow(first_few_rows_2D, origin='lower')


########################################################################################################################
# Clean up
# --------
# Finally lets close and delete the example HDF5 file

h5_file.close()
os.remove(h5_path)

