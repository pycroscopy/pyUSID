# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Chris Smith, Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import sys
import h5py
from sidpy.hdf.reg_ref import *

from .hdf_utils import check_if_main

if sys.version_info.major == 3:
    unicode = str


def copy_region_refs(h5_source, h5_target):
    """
    Check the input dataset for plot groups, copy them if they exist
    Also make references in the Spectroscopic Values and Indices tables

    Parameters
    ----------
    h5_source : HDF5 Dataset
            source dataset to copy references from
    h5_target : HDF5 Dataset
            target dataset the references from h5_source are copied to

    """
    '''
    Check both h5_source and h5_target to ensure that are Main
    '''
    are_main = all([check_if_main(h5_source), check_if_main(h5_target)])
    if not all([isinstance(h5_source, h5py.Dataset), isinstance(h5_target, h5py.Dataset)]):
        raise TypeError('Inputs to copy_region_refs must be HDF5 Datasets')

    # It is OK if objects are in different files

    if are_main:
        h5_source_inds = h5_source.file[h5_source.attrs['Spectroscopic_Indices']]

        h5_spec_inds = h5_target.file[h5_target.attrs['Spectroscopic_Indices']]
        h5_spec_vals = h5_target.file[h5_target.attrs['Spectroscopic_Values']]

    for key in h5_source.attrs.keys():
        if not isinstance(h5_source.attrs[key], h5py.RegionReference):
            continue

        if are_main:
            if h5_source_inds.shape[0] == h5_spec_inds.shape[0]:
                '''
                Spectroscopic dimensions are identical.
                Do direct copy.
                '''
                ref_inds = simple_region_ref_copy(h5_source, h5_target, key)

            else:
                '''
                Spectroscopic dimensions are different.
                Do the dimension reducing copy.
                '''
                ref_inds = copy_reg_ref_reduced_dim(h5_source, h5_target, h5_source_inds, h5_spec_inds, key)

            '''
            Create references for Spectroscopic Indices and Values
            Set the end-point of each hyperslab in the position dimension to the number of
            rows in the index array
            '''

            ref_inds[:, 1, 0][ref_inds[:, 1, 0] > h5_spec_inds.shape[0]] = h5_spec_inds.shape[0] - 1
            spec_inds_ref = create_region_reference(h5_spec_inds, ref_inds)
            h5_spec_inds.attrs[key] = spec_inds_ref
            spec_vals_ref = create_region_reference(h5_spec_vals, ref_inds)
            h5_spec_vals.attrs[key] = spec_vals_ref

        else:
            '''
            If not main datasets, then only simple copy can be used.
            '''
            simple_region_ref_copy(h5_source, h5_target, key)