# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Chris Smith, Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import sys
from warnings import warn
import h5py
import numpy as np

if sys.version_info.major == 3:
    unicode = str

__all__ = ['get_region', 'clean_reg_ref', 'attempt_reg_ref_build',]


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


def clean_reg_ref(h5_dset, reg_ref_tuple, verbose=False):
    """
    Makes sure that the provided instructions for a region reference are indeed valid
    This method has become necessary since h5py allows the writing of region references larger than the maxshape

    Parameters
    ----------
    h5_dset : h5.Dataset instance
        Dataset to which region references will be added as attributes
    reg_ref_tuple : list / tuple
        The slicing information formatted using tuples of slice objects.
    verbose : Boolean (Optional. Default = False)
        Whether or not to print status messages

    Returns
    -------
    is_valid : bool
        Whether or not this
    """
    if not isinstance(reg_ref_tuple, (tuple, dict, slice)):
        raise TypeError('slices should be a tuple, list, or slice but is instead of type '
                        '{}'.format(type(reg_ref_tuple)))
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object but is instead of type '
                        '{}'.format(type(h5_dset)))

    if isinstance(reg_ref_tuple, slice):
        # 1D dataset
        reg_ref_tuple = [reg_ref_tuple]

    if len(reg_ref_tuple) != len(h5_dset.shape):
        raise ValueError('Region reference tuple did not have the same dimensions as the h5 dataset')

    if verbose:
        print('Comparing {} with h5 dataset maxshape of {}'.format(reg_ref_tuple, h5_dset.maxshape))

    new_reg_refs = list()

    for reg_ref_slice, max_size in zip(reg_ref_tuple, h5_dset.maxshape):
        if not isinstance(reg_ref_slice, slice):
            raise TypeError('slices should be a tuple or a list but is instead of type '
                            '{}'.format(type(reg_ref_slice)))

        # For now we will simply make sure that the end of the slice is <= maxshape
        if max_size is not None and reg_ref_slice.stop is not None:
            reg_ref_slice = slice(reg_ref_slice.start, min(reg_ref_slice.stop, max_size), reg_ref_slice.step)

        new_reg_refs.append(reg_ref_slice)

    if verbose:
        print('Region reference tuple now: {}'.format(new_reg_refs))

    return tuple(new_reg_refs)


def attempt_reg_ref_build(h5_dset, dim_names, verbose=False):
    """

    Parameters
    ----------
    h5_dset : h5.Dataset instance
        Dataset to which region references need to be added as attributes
    dim_names : list or tuple
        List of the names of the region references (typically names of dimensions)
    verbose : bool, optional. Default=False
        Whether or not to print debugging statements

    Returns
    -------
    labels_dict : dict
        The slicing information must be formatted using tuples of slice objects.
        For example {'region_1':(slice(None, None), slice (0,1))}
    """
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object but is instead of type '
                        '{}.'.format(type(h5_dset)))
    if not isinstance(dim_names, (list, tuple)):
        raise TypeError('slices should be a list or tuple but is instead of type '
                        '{}'.format(type(dim_names)))

    if len(h5_dset.shape) != 2:
        return dict()

    if not np.all([isinstance(obj, (str, unicode)) for obj in dim_names]):
        raise TypeError('Unable to automatically generate region references for dataset: {} since one or more names'
                        ' of the region references was not a string'.format(h5_dset.name))

    labels_dict = dict()
    if len(dim_names) == h5_dset.shape[0]:
        if verbose:
            print('Most likely a spectroscopic indices / values dataset')
        for dim_index, curr_name in enumerate(dim_names):
            labels_dict[curr_name] = (slice(dim_index, dim_index + 1), slice(None))
    elif len(dim_names) == h5_dset.shape[1]:
        if verbose:
            print('Most likely a position indices / values dataset')
        for dim_index, curr_name in enumerate(dim_names):
            labels_dict[curr_name] = (slice(None), slice(dim_index, dim_index + 1))

    if len(labels_dict) > 0:
        warn('Attempted to automatically build region reference dictionary for dataset: {}.\n'
             'Please specify region references as a tuple of slice objects for each attribute'.format(h5_dset.name))
    else:
        if verbose:
            print('Could not build region references since dataset had shape:{} and number of region references is '
                  '{}'.format(h5_dset.shape, len(dim_names)))
    return labels_dict
