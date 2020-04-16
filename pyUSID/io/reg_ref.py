# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Chris Smith, Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import sys
from collections import Iterable
from warnings import warn
import h5py
import numpy as np

from .write_utils import clean_string_att

if sys.version_info.major == 3:
    unicode = str

__all__ = ['get_region', 'clean_reg_ref', 'attempt_reg_ref_build', 'copy_reg_ref_reduced_dim',
           'create_region_reference', 'get_indices_for_region_ref', 'simple_region_ref_copy', 'write_region_references']


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
    new_reg_refs : tuple
        Instructions for the corrected region reference
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


def get_indices_for_region_ref(h5_main, ref, return_method='slices'):
    """
    Given an hdf5 region reference and the dataset it refers to,
    return an array of indices within that dataset that
    correspond to the reference.

    Parameters
    ----------
    h5_main : HDF5 Dataset
        dataset that the reference can be returned from
    ref : HDF5 Region Reference
        Region reference object
    return_method : {'slices', 'corners', 'points'}
        slices : the reference is return as pairs of slices

        corners : the reference is returned as pairs of corners representing
        the starting and ending indices of each block

        points : the reference is returns as a list of tuples of points

    Returns
    -------
    ref_inds : Numpy Array
        array of indices in the source dataset that ref accesses

    """
    if not isinstance(h5_main, h5py.Dataset):
        raise TypeError('h5_main should be a h5py.Dataset object')
    if not isinstance(ref, h5py.RegionReference):
        raise TypeError('ref should be a h5py.RegionReference object')
    if return_method is not None:
        if not isinstance(return_method, (str, unicode)):
            raise TypeError('return_method should be a string')

    if return_method == 'points':
        def __corners_to_point_array(start, stop):
            """
            Convert a pair of tuples representing two opposite corners of an HDF5 region reference
            into a list of arrays for each dimension.

            Parameters
            ----------
            start : Tuple
                the starting indices of the region
            stop : Tuple
                the final indices of the region

            Returns
            -------
            inds : Tuple of arrays
                the list of points in each dimension

            """
            ranges = []
            for i in range(len(start)):
                if start[i] == stop[i]:
                    ranges.append([stop[i]])
                else:
                    ranges.append(np.arange(start[i], stop[i] + 1, dtype=np.uint))
            grid = np.meshgrid(*ranges, indexing='ij')

            ref_inds = np.asarray(zip(*(x.flat for x in grid)))

            return ref_inds

        return_func = __corners_to_point_array
    elif return_method == 'corners':
        def __corners_to_corners(start, stop):
            return start, stop

        return_func = __corners_to_corners
    elif return_method == 'slices':
        def __corners_to_slices(start, stop):
            """
            Convert a pair of tuples representing two opposite corners of an HDF5 region reference
            into a pair of slices.

            Parameters
            ----------
            start : Tuple
                the starting indices of the region
            stop : Tuple
                the final indices of the region

            Returns
            -------
            slices : list
                pair of slices representing the region

            """
            slices = []
            for idim in range(len(start)):
                slices.append(slice(start[idim], stop[idim]))

            return slices

        return_func = __corners_to_slices

    region = h5py.h5r.get_region(ref, h5_main.id)
    reg_type = region.get_select_type()
    if reg_type == 2:
        """
        Reference is hyperslabs
        """
        ref_inds = []
        for start, end in region.get_select_hyper_blocklist():
            ref_inds.append(return_func(start, end))
        ref_inds = np.array(ref_inds).reshape(-1, len(start))

    elif reg_type == 3:
        """
        Reference is single block
        """
        start, end = region.get_select_bounds()

        ref_inds = return_func(start, end)
    else:
        warn('No method currently exists for converting this type of reference.')
        ref_inds = np.empty(0)

    return ref_inds


def copy_reg_ref_reduced_dim(h5_source, h5_target, h5_source_inds, h5_target_inds, key):
    """
    Copies a region reference from one dataset to another taking into account that a dimension
    has been lost from source to target

    Parameters
    ----------
    h5_source : HDF5 Dataset
            source dataset for region reference copy
    h5_target : HDF5 Dataset
            target dataset for region reference copy
    h5_source_inds : HDF5 Dataset
            indices of each dimension of the h5_source dataset
    h5_target_inds : HDF5 Dataset
            indices of each dimension of the h5_target dataset
    key : String
            Name of attribute in h5_source that contains
            the Region Reference to copy

    Returns
    -------
    ref_inds : Nx2x2 array of unsigned integers
            Array containing pairs of points that define
            the corners of each hyperslab in the region
            reference

    """
    for param, param_name in zip([h5_source, h5_target, h5_source_inds, h5_target_inds],
                                 ['h5_source', 'h5_target', 'h5_source_inds', 'h5_target_inds']):
        if not isinstance(param, h5py.Dataset):
            raise TypeError(param_name + ' should be a h5py.Dataset object')
    if not isinstance(key, (str, unicode)):
        raise TypeError('key should be a string')
    key = key.strip()

    '''
    Determine which dimension is missing from the target
    '''
    lost_dim = []
    for dim in h5_source_inds.attrs['labels']:
        if dim not in h5_target_inds.attrs['labels']:
            lost_dim.append(np.where(h5_source_inds.attrs['labels'] == dim)[0])
    ref = h5_source.attrs[key]
    ref_inds = get_indices_for_region_ref(h5_source, ref, return_method='corners')
    '''
    Convert to proper spectroscopic dimensions
    First is special case for a region reference that spans the entire dataset
    '''
    if len(ref_inds.shape) == 2 and all(ref_inds[0] == [0, 0]) and all(ref_inds[1] + 1 == h5_source.shape):
        ref_inds[1, 1] = h5_target.shape[1] - 1
        ref_inds = np.expand_dims(ref_inds, 0)
    else:
        '''
    More common case of reference made of hyperslabs
        '''
        spec_ind_zeroes = np.where(h5_source_inds[lost_dim] == 0)[1]

        ref_inds = ref_inds.reshape([-1, 2, 2])

        for start, stop in ref_inds[:-1]:
            start[1] = np.where(start[1] == spec_ind_zeroes)[0]
            stop[1] = np.where(stop[1] == spec_ind_zeroes - 1)[0] - 1

        ref_inds[-1, 0, 1] = np.where(ref_inds[-1, 0, 1] == spec_ind_zeroes)[0]
        stop = np.where(ref_inds[-1, 1, 1] == spec_ind_zeroes - 1)[0]
        if stop.size == 0:
            stop = len(spec_ind_zeroes)
        ref_inds[-1, 1, 1] = stop - 1
    '''
    Create the new reference from the indices
    '''
    h5_target.attrs[key] = create_region_reference(h5_target, ref_inds)

    return ref_inds


def create_region_reference(h5_main, ref_inds):
    """
    Create a region reference in the destination dataset using an iterable of pairs of indices
    representing the start and end points of a hyperslab block

    Parameters
    ----------
    h5_main : HDF5 dataset
        dataset the region will be created in
    ref_inds : Iterable
        index pairs, [start indices, final indices] for each block in the
        hyperslab

    Returns
    -------
    new_ref : HDF5 Region reference
        reference in `h5_main` for the blocks of points defined by `ref_inds`

    """
    if not isinstance(h5_main, h5py.Dataset):
        raise TypeError('h5_main should be a h5py.Dataset object')
    if not isinstance(ref_inds, Iterable):
        raise TypeError('ref_inds should be a list or tuple')

    h5_space = h5_main.id.get_space()
    h5_space.select_none()

    for start, stop in ref_inds:
        block = stop - start + 1
        h5_space.select_hyperslab(tuple(start), (1, 1), block=tuple(block), op=1)

    if not h5_space.select_valid():
        warn('Could not create new region reference.')
        return None
    new_ref = h5py.h5r.create(h5_main.id, b'.', h5py.h5r.DATASET_REGION, space=h5_space)

    return new_ref


def simple_region_ref_copy(h5_source, h5_target, key):
    """
    Copies a region reference from one dataset to another
    without alteration

    Parameters
    ----------
    h5_source : HDF5 Dataset
            source dataset for region reference copy
    h5_target : HDF5 Dataset
            target dataset for region reference copy
    key : String
            Name of attribute in h5_source that contains
            the Region Reference to copy

    Returns
    -------
    ref_inds : Nx2x2 array of unsigned integers
            Array containing pairs of points that define
            the corners of each hyperslab in the region
            reference

    """
    for param, param_name in zip([h5_source, h5_target], ['h5_source', 'h5_target']):
        if not isinstance(param, h5py.Dataset):
            raise TypeError(param_name + ' should be a h5py.Dataset object')
    if not isinstance(key, (str, unicode)):
        raise TypeError('key should be a string')

    ref = h5_source.attrs[key]
    ref_inds = get_indices_for_region_ref(h5_source, ref, return_method='corners')
    ref_inds = ref_inds.reshape([-1, 2, 2])
    ref_inds[:, 1, 1] = h5_target.shape[1] - 1
    target_ref = create_region_reference(h5_target, ref_inds)
    h5_target.attrs[key] = target_ref
    return ref_inds


def copy_all_region_refs(h5_source, h5_target):
    """
    Copies only region references from the source dataset to the target dataset

    Parameters
    ----------
    h5_source : h5py.Dataset
        Dataset from which to copy region references
    h5_target : h5py.Dataset
        Dataset to which to copy region references to

    """
    if not isinstance(h5_source, h5py.Dataset):
        raise TypeError("'h5_source' should be a h5py.Dataset object")
    if not isinstance(h5_target, h5py.Dataset):
        raise TypeError("'h5_target' should be a h5py.Dataset object")
    for key in h5_source.attrs.keys():
        if not isinstance(h5_source.attrs[key], h5py.RegionReference):
            continue
        simple_region_ref_copy(h5_source, h5_target, key)


def write_region_references(h5_dset, reg_ref_dict, add_labels_attr=True, verbose=False):
    """
    Creates attributes of a h5py.Dataset that refer to regions in the dataset

    Parameters
    ----------
    h5_dset : h5.Dataset instance
        Dataset to which region references will be added as attributes
    reg_ref_dict : dict
        The slicing information must be formatted using tuples of slice objects.
        For example {'region_1':(slice(None, None), slice (0,1))}
    add_labels_attr : bool, optional, default = True
        Whether or not to write an attribute named 'labels' with the
    verbose : Boolean (Optional. Default = False)
        Whether or not to print status messages
    """
    if not isinstance(reg_ref_dict, dict):
        raise TypeError('slices should be a dictionary but is instead of type '
                        '{}'.format(type(reg_ref_dict)))
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object but is instead of type '
                        '{}'.format(type(h5_dset)))

    if verbose:
        print('Starting to write Region References to Dataset', h5_dset.name, 'of shape:', h5_dset.shape)
    for reg_ref_name, reg_ref_tuple in reg_ref_dict.items():
        if verbose:
            print('About to write region reference:', reg_ref_name, ':', reg_ref_tuple)

        reg_ref_tuple = clean_reg_ref(h5_dset, reg_ref_tuple, verbose=verbose)

        h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]

        if verbose:
            print('Wrote Region Reference:%s' % reg_ref_name)

    '''
    Next, write these label names as an attribute called labels
    Now make an attribute called 'labels' that is a list of strings 
    First ascertain the dimension of the slicing:
    '''
    if add_labels_attr:
        found_dim = False
        dimen_index = None

        for key, val in reg_ref_dict.items():
            if not isinstance(val, (list, tuple)):
                reg_ref_dict[key] = [val]

        for dimen_index, slice_obj in enumerate(list(reg_ref_dict.values())[0]):
            # We make the assumption that checking the start is sufficient
            if slice_obj.start is not None:
                found_dim = True
                break
        if found_dim:
            headers = [None] * len(reg_ref_dict)  # The list that will hold all the names
            for col_name in reg_ref_dict.keys():
                headers[reg_ref_dict[col_name][dimen_index].start] = col_name
            if verbose:
                print('Writing header attributes: {}'.format('labels'))
            # Now write the list of col / row names as an attribute:
            h5_dset.attrs['labels'] = clean_string_att(headers)
        else:
            warn('Unable to write region references for %s' % (h5_dset.name.split('/')[-1]))

        if verbose:
            print('Wrote Region References of Dataset %s' % (h5_dset.name.split('/')[-1]))