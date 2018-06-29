# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Chris Smith, Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import sys
import h5py
import numpy as np

if sys.version_info.major == 3:
    unicode = str

__all__ = []


def get_region(h5_dset, reg_ref_name):
    """
    Gets the region in a dataset specified by a region reference

    Parameters
    ----------
    h5_dset : h5py.Dataset
        Dataset containing the region reference
    reg_ref_name : str / unicode
        Name of the region reference

    Returns
    -------
    value : np.ndarray
        Data specified by the region reference. Note that a squeeze is applied by default.
    """
    if not isinstance(reg_ref_name, (str, unicode)):
        raise TypeError('reg_ref_name should be a string')
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be of type h5py.Dataset')
    # this may raise KeyErrors. Let it
    reg_ref = h5_dset.attrs[reg_ref_name]
    return np.squeeze(h5_dset[reg_ref])