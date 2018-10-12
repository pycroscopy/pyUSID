"""
================================================================================
01. 1D Spectrum or Curve
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
url = 'https://raw.githubusercontent.com/pycroscopy/pyUSID/master/data/AFM_Force_Curve.h5'
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
# Visualize the contents in each of these channels
# ------------------------------------------------
usid.plot_utils.use_nice_plot_params()
for main_dset in usid.hdf_utils.get_all_main(h5_file):
    main_dset.visualize()

########################################################################################################################
# Access the ``Main`` Dataset containing the spectrum of interest
# ---------------------------------------------------------------
# In this case, this is the second channel:
h5_main = usid.hdf_utils.get_all_main(h5_file)[1]
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
# See the contents within the Position Datasets
# --------------------------------------------------
print(h5_main.h5_pos_inds[()], h5_main.h5_pos_vals[()])

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
# Visualize the contents of the Position Indices Dataset
# ------------------------------------------------------
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
for axis, data, title, y_lab in zip(axes.flat,
                                    [h5_main.h5_spec_inds[()].T, h5_main.h5_spec_vals[()].T],
                                    ['Spectroscopic Indices', 'Spectroscopic Values'],
                                    ['Index', h5_main.spec_dim_descriptors[0]]):
    axis.plot(data)
    axis.set_title(title)
    axis.set_xlabel('Row in ' + title)
    axis.set_ylabel(y_lab)

usid.plot_utils.use_scientific_ticks(axis, is_x=False)
fig.suptitle('Ancillary Spectroscopic Datasets', y=1.05)
fig.tight_layout()

########################################################################################################################
# Clean up
# --------
# Finally lets close and delete the example HDF5 file

h5_file.close()
os.remove(h5_path)

