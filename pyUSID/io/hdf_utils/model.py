# -*- coding: utf-8 -*-
"""
Utilities for reading and writing USID datasets that are highly model-dependent (with or without N-dimensional form)

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""
from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import sys
import h5py
import numpy as np
from dask import array as da

from ..dtype_utils import contains_integers, validate_dtype, validate_single_string_arg, validate_string_args, \
    validate_list_of_strings, lazy_load_array

from .base import get_attr, write_simple_attrs, is_editable_h5, write_book_keeping_attrs
from .simple import link_as_main, check_if_main, write_ind_val_dsets, validate_dims_against_main, validate_anc_h5_dsets, copy_dataset
from pyUSID.io.write_utils import validate_dimensions
from ..write_utils import INDICES_DTYPE, make_indices_matrix

if sys.version_info.major == 3:
    unicode = str


def reshape_to_n_dims(h5_main, h5_pos=None, h5_spec=None, get_labels=False, verbose=False, sort_dims=False,
                      lazy=False):
    """
    Reshape the input 2D matrix to be N-dimensions based on the
    position and spectroscopic datasets.

    Parameters
    ----------
    h5_main : HDF5 Dataset
        2D data to be reshaped
    h5_pos : HDF5 Dataset, optional
        Position indices corresponding to rows in `h5_main`
    h5_spec : HDF5 Dataset, optional
        Spectroscopic indices corresponding to columns in `h5_main`
    get_labels : bool, optional
        Whether or not to return the dimension labels.  Default False
    verbose : bool, optional
        Whether or not to print debugging statements
    sort_dims : bool
        If True, the data is sorted so that the dimensions are in order from slowest to fastest
        If False, the data is kept in the original order
        If `get_labels` is also True, the labels are sorted as well.
    lazy : bool, optional. Default = False
        If False, ds_Nd will be a numpy.ndarray object - this is suitable if the HDF5 dataset fits into memory
        If True, ds_Nd will be a dask.array object - This is suitable if the HDF5 dataset is too large to fit into
        memory. Note that this will bea lazy computation meaning that the returned object just contains the instructions
        . In order to get the actual value or content in numpy arrays, call ds_Nd.compute()

    Returns
    -------
    ds_Nd : N-D numpy array or dask.array object
        N dimensional array arranged as [positions slowest to fastest, spectroscopic slowest to fastest]
    success : boolean or string
        True if full reshape was successful

        "Positions" if it was only possible to reshape by
        the position dimensions

        False if no reshape was possible
    ds_labels : list of str
        List of the labels of each dimension of `ds_Nd`

    Notes
    -----
    If either `h5_pos` or `h5_spec` are not provided, the function will first
    attempt to find them as attributes of `h5_main`.  If that fails, it will
    generate dummy values for them.

    """
    # TODO: automatically switch on lazy if the data is larger than memory
    # TODO: sort_dims does not appear to do much. Functions as though it was always True

    if h5_pos is None and h5_spec is None:
        if not check_if_main(h5_main):
            raise ValueError('if h5_main is a h5py.Dataset it should be a Main dataset')
    else:
        if not isinstance(h5_main, (h5py.Dataset, np.ndarray, da.core.Array)):
            raise TypeError('h5_main should either be a h5py.Dataset or numpy array')

    if h5_pos is not None:
        if not isinstance(h5_pos, (h5py.Dataset, np.ndarray, da.core.Array)):
            raise TypeError('h5_pos should either be a h5py.Dataset or numpy array')
        if h5_pos.shape[0] != h5_main.shape[0]:
            raise ValueError('The size of h5_pos: {} does not match with h5_main: {}'.format(h5_pos.shape,
                                                                                             h5_main.shape))

    if h5_spec is not None:
        if not isinstance(h5_spec, (h5py.Dataset, np.ndarray, da.core.Array)):
            raise TypeError('h5_spec should either be a h5py.Dataset or numpy array')
        if h5_spec.shape[1] != h5_main.shape[1]:
            raise ValueError('The size of h5_spec: {} does not match with h5_main: {}'.format(h5_spec.shape,
                                                                                              h5_main.shape))

    pos_labs = np.array(['Positions'])
    spec_labs = np.array(['Spectral_Step'])
    if h5_pos is None:
        """
        Get the Position datasets from the references if possible
        """
        if isinstance(h5_main, h5py.Dataset):
            try:
                h5_pos = h5_main.file[h5_main.attrs['Position_Indices']]
                ds_pos = h5_pos[()]
                pos_labs = get_attr(h5_pos, 'labels')
            except KeyError:
                print('No position datasets found as attributes of {}'.format(h5_main.name))
                if len(h5_main.shape) > 1:
                    ds_pos = np.arange(h5_main.shape[0], dtype=INDICES_DTYPE).reshape(-1, 1)
                    pos_labs = np.array(['Position Dimension {}'.format(ipos) for ipos in range(ds_pos.shape[1])])
                else:
                    ds_pos = np.array(0, dtype=INDICES_DTYPE).reshape(-1, 1)
        else:
            ds_pos = np.arange(h5_main.shape[0], dtype=INDICES_DTYPE).reshape(-1, 1)
            pos_labs = np.array(['Position Dimension {}'.format(ipos) for ipos in range(ds_pos.shape[1])])
    elif isinstance(h5_pos, h5py.Dataset):
        """
    Position Indices dataset was provided
        """
        ds_pos = h5_pos[()]
        pos_labs = get_attr(h5_pos, 'labels')
    elif isinstance(h5_pos, (np.ndarray, da.core.Array)):
        ds_pos = np.atleast_2d(h5_pos)
        pos_labs = np.array(['Position Dimension {}'.format(ipos) for ipos in range(ds_pos.shape[1])])
    else:
        raise TypeError('Position Indices must be either h5py.Dataset or None')

    if h5_spec is None:
        """
        Get the Spectroscopic datasets from the references if possible
        """
        if isinstance(h5_main, h5py.Dataset):
            try:
                h5_spec = h5_main.file[h5_main.attrs['Spectroscopic_Indices']]
                ds_spec = h5_spec[()]
                spec_labs = get_attr(h5_spec, 'labels')
            except KeyError:
                print('No spectroscopic datasets found as attributes of {}'.format(h5_main.name))
                if len(h5_main.shape) > 1:
                    ds_spec = np.arange(h5_main.shape[1], dtype=INDICES_DTYPE).reshape([1, -1])
                    spec_labs = np.array(['Spectral Dimension {}'.format(ispec) for ispec in range(ds_spec.shape[0])])
                else:
                    ds_spec = np.array(0, dtype=INDICES_DTYPE).reshape([1, 1])
        else:
            ds_spec = np.arange(h5_main.shape[1], dtype=INDICES_DTYPE).reshape([1, -1])
            spec_labs = np.array(['Spectral Dimension {}'.format(ispec) for ispec in range(ds_spec.shape[0])])

    elif isinstance(h5_spec, h5py.Dataset):
        """
    Spectroscopic Indices dataset was provided
        """
        ds_spec = h5_spec[()]
        spec_labs = get_attr(h5_spec, 'labels')
    elif isinstance(h5_spec, (np.ndarray, da.core.Array)):
        ds_spec = h5_spec
        spec_labs = np.array(['Spectral Dimension {}'.format(ispec) for ispec in range(ds_spec.shape[0])])
    else:
        raise TypeError('Spectroscopic Indices must be either h5py.Dataset or None')

    '''
    Sort the indices from fastest to slowest
    '''
    pos_sort = get_sort_order(np.transpose(ds_pos))
    spec_sort = get_sort_order(ds_spec)

    if verbose:
        print('Position dimensions:', pos_labs)
        print('Position sort order:', pos_sort)
        print('Spectroscopic Dimensions:', spec_labs)
        print('Spectroscopic sort order:', spec_sort)

    '''
    Get the size of each dimension in the sorted order
    '''
    pos_dims = get_dimensionality(np.transpose(ds_pos), pos_sort)
    spec_dims = get_dimensionality(ds_spec, spec_sort)

    if np.prod(pos_dims) != h5_main.shape[0]:
        mesg = 'Product of position dimension sizes: {} = {} not matching ' \
               'with size of first axis of main dataset: {}. One or more ' \
               'dimensions are dependent dimensions and not marked as such' \
               '.'.format(pos_dims, np.prod(pos_dims), h5_main.shape[0])
        raise ValueError(mesg)
    if np.prod(spec_dims) != h5_main.shape[1]:
        mesg = 'Product of spectroscopic dimension sizes: {} = {} not matching ' \
               'with size of second axis of main dataset: {}. One or more ' \
               'dimensions are dependent dimensions and not marked as such' \
               '.'.format(spec_dims, np.prod(spec_dims), h5_main.shape[1])
        raise ValueError(mesg)

    if verbose:
        print('\nPosition dimensions (sort applied):', pos_labs[pos_sort])
        print('Position dimensionality (sort applied):', pos_dims)
        print('Spectroscopic dimensions (sort applied):', spec_labs[spec_sort])
        print('Spectroscopic dimensionality (sort applied):', spec_dims)

    if lazy:
        ds_main = lazy_load_array(h5_main)
    else:
        ds_main = h5_main[()]

    """
    Now we reshape the dataset based on those dimensions
    numpy reshapes correctly when the dimensions are arranged from slowest to fastest. 
    Since the sort orders we have are from fastest to slowest, we need to reverse the orders
    for both the position and spectroscopic dimensions
    """
    if verbose:
        print('Will attempt to reshape main dataset from:\n{} to {}'.format(ds_main.shape, pos_dims[::-1] + spec_dims[::-1]))

    try:
        ds_Nd = ds_main.reshape(pos_dims[::-1] + spec_dims[::-1])

    except ValueError:
        warn('Could not reshape dataset to full N-dimensional form.  Attempting reshape based on position only.')
        try:
            ds_Nd = ds_main.reshape(pos_dims[::-1] + [-1])

        except ValueError:
            warn('Reshape by position only also failed.  Will keep dataset in 2d form.')
            if get_labels:
                return ds_main, False, ['Position', 'Spectral Step']
            else:
                return ds_main, False

        # No exception
        else:
            if get_labels:
                return ds_Nd, 'Positions', ['Position'] + spec_labs
            else:
                return ds_Nd, 'Positions'

    all_labels = np.hstack((pos_labs[pos_sort][::-1],
                            spec_labs[spec_sort][::-1]))

    if verbose:
        print('\nAfter reshaping, labels are', all_labels)
        print('Data shape is', ds_Nd.shape)

    """
    At this point, the data is arranged from slowest to fastest dimension in both pos and spec
    """
    if sort_dims:
        results = [ds_Nd, True]
        if get_labels:
            results.append(all_labels)
        return results

    if verbose:
        print('\nGoing to put dimensions back in the same order as in the file:')

    swap_axes = list()
    # Compare the original order of the pos / spec labels with where these dimensions occur in the sorted labels
    for lab in pos_labs:
        swap_axes.append(np.argwhere(all_labels == lab).squeeze())
    for lab in spec_labs:
        swap_axes.append(np.argwhere(all_labels == lab).squeeze())

    swap_axes = np.array(swap_axes)

    if verbose:
        print('Axes will permuted in this order:', swap_axes)
        print('New labels ordering:', all_labels[swap_axes])

    ds_Nd = ds_Nd.transpose(tuple(swap_axes))
    results = [ds_Nd, True]

    if verbose:
        print('Dataset now of shape:', ds_Nd.shape)

    if get_labels:
        '''
        Get the labels in the proper order
        '''
        results.append(all_labels[swap_axes])

    return results


def reshape_from_n_dims(data_n_dim, h5_pos=None, h5_spec=None, verbose=False):
    """
    Reshape the input 2D matrix to be N-dimensions based on the
    position and spectroscopic datasets.

    Parameters
    ----------
    data_n_dim : numpy.array or dask.array.core.Array
        N dimensional array arranged as [positions dimensions..., spectroscopic dimensions]
        If h5_pos and h5_spec are not provided, this function will have to assume that the dimensions
        are arranged as [positions slowest to fastest, spectroscopic slowest to fastest].
        This restriction is removed if h5_pos and h5_spec are provided
    h5_pos : HDF5 Dataset, numpy.array or dask.array.core.Array
        Position indices corresponding to rows in the final 2d array
        The dimensions should be arranged in terms of rate of change corresponding to data_n_dim.
        In other words if data_n_dim had two position dimensions arranged as [pos_fast, pos_slow, spec_dim_1....],
        h5_pos should be arranged as [pos_fast, pos_slow]
    h5_spec : HDF5 Dataset, numpy. array or dask.array.core.Array
        Spectroscopic indices corresponding to columns in the final 2d array
        The dimensions should be arranged in terms of rate of change corresponding to data_n_dim.
        In other words if data_n_dim had two spectral dimensions arranged as [pos_dim_1,..., spec_fast, spec_slow],
        h5_spec should be arranged as [pos_slow, pos_fast]
    verbose : bool, optional. Default = False
        Whether or not to print log statements

    Returns
    -------
    ds_2d : numpy.array
        2 dimensional numpy array arranged as [positions, spectroscopic]
    success : boolean or string
        True if full reshape was successful

        "Positions" if it was only possible to reshape by
        the position dimensions

        False if no reshape was possible

    Notes
    -----
    If either `h5_pos` or `h5_spec` are not provided, the function will
    assume the first dimension is position and the remaining are spectroscopic already
    in order from fastest to slowest.

    """
    if not isinstance(data_n_dim, (np.ndarray, da.core.Array)):
        raise TypeError('data_n_dim is not a numpy or dask array')

    if h5_spec is None and h5_pos is None:
        raise ValueError('at least one of h5_pos or h5_spec must be specified for an attempt to reshape to 2D')

    if data_n_dim.ndim < 2:
        return data_n_dim, True

    if h5_pos is None:
        pass
    elif isinstance(h5_pos, h5py.Dataset):
        '''
        Position Indices dataset was provided
        '''
        ds_pos = h5_pos[()]
    elif isinstance(h5_pos, da.core.Array):
        ds_pos = h5_pos.compute()

    elif isinstance(h5_pos, np.ndarray):
        ds_pos = h5_pos
    else:
        raise TypeError('Position Indices must be either h5py.Dataset or None')

    if h5_spec is None:
        pass
    elif isinstance(h5_spec, h5py.Dataset):
        '''
        Spectroscopic Indices dataset was provided
        '''
        ds_spec = h5_spec[()]
    elif isinstance(h5_spec, da.core.Array):
        ds_spec = h5_spec.compute()
    elif isinstance(h5_spec, np.ndarray):
        ds_spec = h5_spec
    else:
        raise TypeError('Spectroscopic Indices must be either h5py.Dataset or None')

    if h5_spec is None and h5_pos is not None:
        if verbose:
            print('Spectral indices not provided but position indices provided.\n'
                  'Building spectral indices assuming that dimensions are arranged as slow -> fast')
        pos_dims = get_dimensionality(ds_pos, index_sort=get_sort_order(ds_pos))
        if not np.all([x in data_n_dim.shape for x in pos_dims]):
            raise ValueError('Dimension sizes in pos_dims: {} do not exist in data_n_dim shape: '
                             '{}'.format(pos_dims, data_n_dim.shape))
        spec_dims = [col for col in list(data_n_dim.shape[len(pos_dims):])]
        if verbose:
            print('data has dimensions: {}. Provided position indices had dimensions of size: {}. Spectral dimensions '
                  'will built with dimensions: {}'.format(data_n_dim.shape, pos_dims, spec_dims))
        ds_spec = make_indices_matrix(spec_dims, is_position=False)

    elif h5_pos is None and h5_spec is not None:
        if verbose:
            print('Position indices not provided but spectral indices provided.\n'
                  'Building position indices assuming that dimensions are arranged as slow -> fast')
        spec_dims = get_dimensionality(ds_spec, index_sort=get_sort_order(ds_spec))
        if not np.all([x in data_n_dim.shape for x in spec_dims]):
            raise ValueError('Dimension sizes in spec_dims: {} do not exist in data_n_dim shape: '
                             '{}'.format(spec_dims, data_n_dim.shape))
        pos_dims = [col for col in list(data_n_dim.shape[:data_n_dim.ndim-len(spec_dims)])]
        if verbose:
            print('data has dimensions: {}. Spectroscopic position indices had dimensions of size: {}. Position '
                  'dimensions will built with dimensions: {}'.format(data_n_dim.shape, spec_dims, pos_dims))
        ds_pos = make_indices_matrix(pos_dims, is_position=True)

    elif h5_spec is not None and h5_pos is not None:
        if ds_pos.shape[0] * ds_spec.shape[1] != np.product(data_n_dim.shape):
            raise ValueError('The product ({}) of the number of positions ({}) and spectroscopic ({}) observations is '
                             'not equal to the product ({}) of the data shape ({})'
                             '.'.format(ds_pos.shape[0] * ds_spec.shape[1], ds_pos.shape[0], ds_spec.shape[1],
                                        np.product(data_n_dim.shape), data_n_dim.shape))

        if ds_pos.shape[1] + ds_spec.shape[0] != data_n_dim.ndim:
            # This may mean that the dummy position or spectroscopic axes has been squeezed out!
            # Dask does NOT allow singular dimensions apparently. So cannot do expand_dims. Handle later

            if ds_pos.size == 1 or ds_spec.size == 1:
                if verbose:
                    print('ALL Position dimensions squeezed: {}. ALL Spectroscopic dimensions squeezed: {}'
                          '.'.format(ds_pos.size == 1, ds_spec.size == 1))
            else:
                raise ValueError('The number of position ({}) and spectroscopic ({}) dimensions do not match with the '
                                 'dimensionality of the N-dimensional dataset: {}'
                                 '.'.format(ds_pos.shape[1], ds_spec.shape[0], data_n_dim.ndim))

    '''
    Sort the indices from fastest to slowest
    '''
    if ds_pos.size == 1:
        # Position dimension squeezed out:
        pos_sort = []
    else:
        pos_sort = get_sort_order(np.transpose(ds_pos))

    if ds_spec.size == 1:
        # Spectroscopic axis squeezed out:
        spec_sort = []
    else:
        spec_sort = get_sort_order(ds_spec)

    if h5_spec is None:
        spec_sort = spec_sort[::-1]
    if h5_pos is None:
        pos_sort = pos_sort[::-1]

    if verbose:
        print('Position sort order: {}'.format(pos_sort))
        print('Spectroscopic sort order: {}'.format(spec_sort))

    '''
    Now we transpose the axes associated with the spectroscopic dimensions
    so that they are in the same order as in the index array
    '''
    swap_axes = np.uint16(np.append(pos_sort[::-1], spec_sort[::-1] + len(pos_sort)))

    if verbose:
        print('swap axes: {} to be applied to N dimensional data of shape {}'.format(swap_axes, data_n_dim.shape))

    data_n_dim_2 = data_n_dim.transpose(tuple(swap_axes))

    if verbose:
        print('N dimensional data shape after axes swap: {}'.format(data_n_dim_2.shape))

    '''
    Now we reshape the dataset based on those dimensions
    We must use the spectroscopic dimensions in reverse order
    '''
    try:
        ds_2d = data_n_dim_2.reshape([ds_pos.shape[0], ds_spec.shape[1]])
    except ValueError:
        raise ValueError('Could not reshape dataset to full N-dimensional form')

    return ds_2d, True


def get_dimensionality(ds_index, index_sort=None):
    """
    Get the size of each index dimension in a specified sort order

    Parameters
    ----------
    ds_index : 2D HDF5 Dataset or numpy array
        Row matrix of indices
    index_sort : Iterable of unsigned integers (Optional)
        Sort that can be applied to dimensionality.
        For example - Order of rows sorted from fastest to slowest

    Returns
    -------
    sorted_dims : list of unsigned integers
        Dimensionality of each row in ds_index.  If index_sort is supplied, it will be in the sorted order

    """
    if isinstance(ds_index, da.core.Array):
        ds_index = ds_index.compute()
    if not isinstance(ds_index, (np.ndarray, h5py.Dataset)):
        raise TypeError('ds_index should either be a numpy array or h5py.Dataset')

    if ds_index.shape[0] > ds_index.shape[1]:
        # must be spectroscopic like in shape (few rows, more cols)
        ds_index = np.transpose(ds_index)

    if index_sort is None:
        index_sort = np.arange(ds_index.shape[0])
    else:
        if not contains_integers(index_sort, min_val=0):
            raise ValueError('index_sort should contain integers > 0')
        index_sort = np.array(index_sort)
        if index_sort.ndim != 1:
            raise ValueError('index_sort should be a 1D array')
        if len(np.unique(index_sort)) > ds_index.shape[0]:
            raise ValueError('length of index_sort ({}) should be smaller than number of dimensions in provided dataset'
                             ' ({}'.format(len(np.unique(index_sort)), ds_index.shape[0]))
        if set(np.arange(ds_index.shape[0])) !=  set(index_sort):
            raise ValueError('Sort order of dimensions ({}) not  matching  with number of dimensions ({})'
                             ''.format(index_sort, ds_index.shape[0]))

    sorted_dims = [len(np.unique(row)) for row in np.array(ds_index, ndmin=2)[index_sort]]
    return sorted_dims


def get_sort_order(ds_spec):
    """
    Find how quickly the spectroscopic values are changing in each row
    and the order of rows from fastest changing to slowest.

    Parameters
    ----------
    ds_spec : 2D HDF5 dataset or numpy array
        Rows of indices to be sorted from fastest changing to slowest

    Returns
    -------
    change_sort : List of unsigned integers
        Order of rows sorted from fastest changing to slowest

    """
    if isinstance(ds_spec, da.core.Array):
        ds_spec = ds_spec.compute()
    if not isinstance(ds_spec, (np.ndarray, h5py.Dataset)):
        raise TypeError('ds_spec should either be a numpy array or h5py.Dataset')

    if ds_spec.shape[0] > ds_spec.shape[1]:
        # must be spectroscopic like in shape (few rows, more cols)
        ds_spec = np.transpose(ds_spec)

    change_count = [len(np.where([row[i] != row[i - 1] for i in range(len(row))])[0]) for row in ds_spec]
    change_sort = np.argsort(change_count)[::-1]

    return change_sort


def get_unit_values(ds_inds, ds_vals, dim_names=None, all_dim_names=None, is_spec=None, verbose=False):
    """
    Gets the unit arrays of values that describe the spectroscopic dimensions

    Parameters
    ----------
    ds_inds : h5py.Dataset or numpy.ndarray
        Spectroscopic or Position Indices dataset
    ds_vals : h5py.Dataset or numpy.ndarray
        Spectroscopic or Position Values dataset
    dim_names : str, or list of str, Optional
        Names of the dimensions of interest. Default = all
    all_dim_names : list of str, Optional
        Names of all the dimensions in these datasets. Use this if supplying numpy arrays instead of h5py.Dataset
        objects for h5_inds, h5_vals since there is no other way of getting the dimension names.
    is_spec : bool, optional
        Whether or not the provided ancillary datasets are position or spectroscopic
        The user is recommended to supply this parameter whenever it is known
        By default, this function will attempt to recognize the answer based on the shape of the datasets.
    verbose : bool, optional
        Whether or not to print debugging statements. Default - off

    Note - this function can be extended / modified for ancillary position dimensions as well

    Returns
    -------
    unit_values : dict
        Dictionary containing the unit array for each dimension. The name of the dimensions are the keys.

    """
    if all_dim_names is None:
        allowed_types = h5py.Dataset
    else:
        all_dim_names = validate_list_of_strings(all_dim_names, 'all_dim_names')
        all_dim_names = np.array(all_dim_names)
        allowed_types = (h5py.Dataset, np.ndarray)

    for dset, dset_name in zip([ds_inds, ds_vals], ['ds_inds', 'ds_vals']):
        if not isinstance(dset, allowed_types):
            raise TypeError(dset_name + ' should be of type: {}'.format(allowed_types))

    # For now, we will throw an error if even a single dimension is listed as an incomplete dimension:
    if isinstance(ds_inds, h5py.Dataset):
        if np.any(['incomplete_dimensions' in dset.attrs.keys() for dset in [ds_inds, ds_vals]]):
            try:
                incomp_dims_inds = get_attr(ds_inds, 'incomplete_dimensions')
            except KeyError:
                incomp_dims_inds = None
            try:
                incomp_dims_vals = get_attr(ds_vals, 'incomplete_dimensions')
            except KeyError:
                incomp_dims_vals = None
            if incomp_dims_inds is None and incomp_dims_vals is not None:
                incomp_dims = incomp_dims_vals
            elif incomp_dims_inds is not None and incomp_dims_vals is None:
                incomp_dims = incomp_dims_inds
            else:
                # ensure that both attributes are the same
                if incomp_dims_vals != incomp_dims_inds:
                    raise ValueError('Provided indices ({}) and values ({}) datasets were marked with different values '
                                     'for incomplete_datasets.'.format(incomp_dims_inds, incomp_dims_vals))
                incomp_dims = incomp_dims_vals

            all_dim_names = get_attr(ds_inds, 'labels')
            raise ValueError('Among all dimensions: {}, These dimensions were marked as incomplete dimensions: {}'
                             '. You are recommended to find unit values manually'.format(all_dim_names, incomp_dims))

    # Do we need to check that the provided inds and vals correspond to the same main dataset?
    if ds_inds.shape != ds_vals.shape:
        raise ValueError('h5_inds: {} and h5_vals: {} should have the same shapes'.format(ds_inds.shape, ds_vals.shape))

    if all_dim_names is None:
        all_dim_names = get_attr(ds_inds, 'labels')
    if verbose:
        print('All dimensions: {}'.format(all_dim_names))

    # First load to memory
    inds_mat = ds_inds[()]
    vals_mat = ds_vals[()]

    if is_spec is None:
        # Attempt to recognize the type automatically
        is_spec = False
        if inds_mat.shape[0] < inds_mat.shape[1]:
            is_spec = True
    else:
        if not isinstance(is_spec, bool):
            raise TypeError('is_spec should be a boolean. Provided object is of type: {}'.format(type(is_spec)))

    if verbose:
        print(
            'Ancillary matrices of shape: {}, hence determined to be Spectroscopic:{}'.format(inds_mat.shape, is_spec))

    if not is_spec:
        # Convert to spectral shape
        inds_mat = np.transpose(inds_mat)
        vals_mat = np.transpose(vals_mat)

    if len(all_dim_names) != inds_mat.shape[0]:
        raise ValueError('Length of dimension names list: {} not matching with shape of dataset: {}'
                         '.'.format(len(all_dim_names), inds_mat.shape[0]))

    if dim_names is None:
        dim_names = all_dim_names
        if verbose:
            print('Going to return unit values for all dimensions: {}'.format(all_dim_names))
    else:
        dim_names = validate_list_of_strings(dim_names, 'dim_names')

        if verbose:
            print('Checking to make sure that the target dimension names: {} exist in the datasets attributes: {}'
                  '.'.format(dim_names, all_dim_names))

        # check to make sure that the dimension names exist in the datasets:
        for dim_name in dim_names:
            if dim_name not in all_dim_names:
                raise KeyError('Dimension {} does not exist in the provided ancillary datasets'.format(dim_name))

    unit_values = dict()
    for dim_name in all_dim_names:
        # Find the row in the spectroscopic indices that corresponds to the dimensions we want to slice:
        if verbose:
            print('Looking for dimension: {} in {}'.format(dim_name, dim_names))
        desired_row_ind = np.where(all_dim_names == dim_name)[0][0]

        inds_for_dim = inds_mat[desired_row_ind]
        # Wherever this dimension goes to 0 - start of a new tile
        starts = np.where(inds_for_dim == np.min(inds_for_dim))[0]
        if starts[0] != 0:
            raise ValueError('Spectroscopic Indices for dimension: "{}" not '
                             'starting with 0. Please fix this and try again'
                             '.'.format(dim_name))

        # There may be repetitions in addition to tiling. Find how the the positions increase.
        # 1 = repetition, > 1 = new tile
        step_sizes = np.hstack(([1], np.diff(starts)))
        # This array is of the same length as the full indices array

        # We should expect only two values of step sizes for a regular dimension (tiles of the same size):
        # 1 for same value repeating and a big jump in indices when the next tile starts
        # If the repeats / tiles are of different lengths, then this is not a regular dimension.
        # What does a Unit Values vector even mean in this case? Just raise an error for now
        if np.where(np.unique(step_sizes) - 1)[0].size > 1:
            raise ValueError('Non constant step sizes')

        # Finding Start of a new tile
        tile_starts = np.where(step_sizes > 1)[0]

        # converting these indices to correct indices that can be mapped straight to
        if len(tile_starts) < 1:
            # Dimension(s) with no tiling at all
            # Make it look as though the next tile starts at the end of the whole indices vector
            tile_starts = np.array([0, len(inds_for_dim)])
        else:
            # Dimension with some form of repetition
            tile_starts = np.hstack(([0], starts[tile_starts]))

            # Verify that each tile is identical here
            # Last tile will not be checked unless we add the length of the indices vector as the start of next tile
            tile_starts = np.hstack((tile_starts, [len(inds_for_dim)]))
            subsections = [inds_for_dim[tile_starts[ind]: tile_starts[ind + 1]] for ind in range(len(tile_starts) - 1)]
            if np.max(np.diff(subsections, axis=0)) != 0:
                # Should get unit values for ALL dimensions regardless of expectations to catch such scenarios.
                raise ValueError('Values in each tile of dimension: {} are different'.format(dim_name))

        # Now looking within the first tile:
        subsection = inds_for_dim[tile_starts[0]:tile_starts[1]]
        # remove all repetitions. ie - take indices only where jump == 1
        step_inds = np.hstack(([0], np.where(np.hstack(([0], np.diff(subsection))))[0]))
        # Finally, use these indices to get the values
        if dim_name in dim_names:
            # Only add this dimension to dictionary if requwested.
            unit_values[dim_name] = vals_mat[desired_row_ind, step_inds]

    return unit_values


def write_main_dataset(h5_parent_group, main_data, main_data_name, quantity, units, pos_dims, spec_dims,
                       main_dset_attrs=None, h5_pos_inds=None, h5_pos_vals=None, h5_spec_inds=None, h5_spec_vals=None,
                       aux_spec_prefix='Spectroscopic_', aux_pos_prefix='Position_', verbose=False,
                       slow_to_fast=False, **kwargs):
    """
    Writes the provided data as a 'Main' dataset with all appropriate linking.
    By default, the instructions for generating the ancillary datasets should be specified using the pos_dims and
    spec_dims arguments as dictionary objects. Alternatively, if both the indices and values datasets are already
    available for either/or the positions / spectroscopic, they can be specified using the keyword arguments. In this
    case, fresh datasets will not be generated.

    Parameters
    ----------
    h5_parent_group : :class:`h5py.Group`
        Parent group under which the datasets will be created
    main_data : numpy.ndarray, dask.array.core.Array, list or tuple
        2D matrix formatted as [position, spectral] or a list / tuple with the shape for an empty dataset.
        If creating an empty dataset - the dtype must be specified via a kwarg.
    main_data_name : String / Unicode
        Name to give to the main dataset. This cannot contain the '-' character.
    quantity : String / Unicode
        Name of the physical quantity stored in the dataset. Example - 'Current'
    units : String / Unicode
        Name of units for the quantity stored in the dataset. Example - 'A' for amperes
    pos_dims : Dimension or array-like of Dimension objects
        Sequence of Dimension objects that provides all necessary instructions for constructing the indices and values
        datasets
        Object specifying the instructions necessary for building the Position indices and values datasets
    spec_dims : Dimension or array-like of Dimension objects
        Sequence of Dimension objects that provides all necessary instructions for constructing the indices and values
        datasets
        Object specifying the instructions necessary for building the Spectroscopic indices and values datasets
    main_dset_attrs : dictionary, Optional
        Dictionary of parameters that will be written to the main dataset. Do NOT include region references here.
    h5_pos_inds : h5py.Dataset, Optional
        Dataset that will be linked with the name "Position_Indices"
    h5_pos_vals : h5py.Dataset, Optional
        Dataset that will be linked with the name "Position_Values"
    h5_spec_inds : h5py.Dataset, Optional
        Dataset that will be linked with the name "Spectroscopic_Indices"
    h5_spec_vals : h5py.Dataset, Optional
        Dataset that will be linked with the name "Spectroscopic_Values"
    aux_spec_prefix : str or unicode, Optional
        Default prefix for Spectroscopic datasets. Default = "Spectroscopic"
    aux_pos_prefix : str or unicode, Optional
        Default prefix for Position datasets. Default = "Position"
    verbose : bool, Optional, default=False
        If set to true - prints debugging logs
    slow_to_fast : bool, Optional. Default=False
        Set to True if the dimensions are arranged from slowest varying to fastest varying.
        Set to False otherwise.
    kwargs will be passed onto the creation of the dataset. Please pass chunking, compression, dtype, and other
        arguments this way

    Returns
    -------
    h5_main : USIDataset
        Reference to the main dataset

    """
    def __check_anc_before_creation(aux_prefix, dim_type='pos'):
        aux_prefix = validate_single_string_arg(aux_prefix, 'aux_' + dim_type + '_prefix')
        if not aux_prefix.endswith('_'):
            aux_prefix += '_'
        if '-' in aux_prefix:
            warn('aux_' + dim_type + ' should not contain the "-" character. Reformatted name from:{} to '
                                     '{}'.format(aux_prefix, aux_prefix.replace('-', '_')))
        aux_prefix = aux_prefix.replace('-', '_')
        for dset_name in [aux_prefix + 'Indices', aux_prefix + 'Values']:
            if dset_name in h5_parent_group.keys():
                # TODO: What if the contained data was correct?
                raise KeyError('Dataset named: ' + dset_name + ' already exists in group: '
                                                               '{}. Consider passing these datasets using kwargs (if they are correct) instead of providing the pos_dims and spec_dims arguments'.format(h5_parent_group.name))
        return aux_prefix

    def __ensure_anc_in_correct_file(h5_inds, h5_vals, prefix):
        if h5_inds.file != h5_vals.file:
            raise ValueError('Provided ' + prefix + ' datasets are present in different HDF5 files!')

        if h5_inds.file != h5_parent_group.file:
            # Need to copy over the anc datasets to the new group
            if verbose:
                print('Need to copy over ancillary datasets: {} and {} to '
                      'destination group: {} which is in a different HDF5 '
                      'file'.format(h5_inds, h5_vals, h5_parent_group))
            ret_vals = [copy_dataset(x, h5_parent_group, verbose=verbose) for x in [h5_inds, h5_vals]]
        else:
            ret_vals = [h5_inds, h5_vals]
        return tuple(ret_vals)

    if not isinstance(h5_parent_group, (h5py.Group, h5py.File)):
        raise TypeError('h5_parent_group should be a h5py.File or h5py.Group object')
    if not is_editable_h5(h5_parent_group):
        raise ValueError('The provided file is not editable')
    if verbose:
        print('h5 group and file OK')

    quantity, units, main_data_name = validate_string_args([quantity, units, main_data_name],
                                                           ['quantity', 'units', 'main_data_name'])
    if verbose:
        print('quantity, units, main_data_name all OK')

    quantity = quantity.strip()
    units = units.strip()
    main_data_name = main_data_name.strip()
    if '-' in main_data_name:
        warn('main_data_name should not contain the "-" character. Reformatted name from:{} to '
             '{}'.format(main_data_name, main_data_name.replace('-', '_')))
    main_data_name = main_data_name.replace('-', '_')

    if isinstance(main_data, (list, tuple)):
        if not contains_integers(main_data, min_val=1):
            raise ValueError('main_data if specified as a shape should be a list / tuple of integers >= 1')
        if len(main_data) != 2:
            raise ValueError('main_data if specified as a shape should contain 2 numbers')
        if 'dtype' not in kwargs:
            raise ValueError('dtype must be included as a kwarg when creating an empty dataset')
        _ = validate_dtype(kwargs.get('dtype'))
        main_shape = main_data
        if verbose:
            print('Selected empty dataset creation. OK so far')
    elif isinstance(main_data, (np.ndarray, da.core.Array)):
        if main_data.ndim != 2:
            raise ValueError('main_data should be a 2D array')
        main_shape = main_data.shape
        if verbose:
            print('Provided numpy or Dask array for main_data OK so far')
    else:
        raise TypeError('main_data should either be a numpy array or a tuple / list with the shape of the data')

    if h5_pos_inds is not None and h5_pos_vals is not None:
        # The provided datasets override fresh building instructions.
        validate_anc_h5_dsets(h5_pos_inds, h5_pos_vals, main_shape, is_spectroscopic=False)
        if verbose:
            print('The shapes of the provided h5 position indices and values are OK')
        h5_pos_inds, h5_pos_vals = __ensure_anc_in_correct_file(h5_pos_inds, h5_pos_vals, 'Position')
    else:
        aux_pos_prefix = __check_anc_before_creation(aux_pos_prefix, dim_type='pos')
        pos_dims = validate_dimensions(pos_dims, dim_type='Position')
        validate_dims_against_main(main_shape, pos_dims, is_spectroscopic=False)
        if verbose:
            print('Passed all pre-tests for creating position datasets')
        h5_pos_inds, h5_pos_vals = write_ind_val_dsets(h5_parent_group, pos_dims, is_spectral=False, verbose=verbose,
                                                       slow_to_fast=slow_to_fast, base_name=aux_pos_prefix)
        if verbose:
            print('Created position datasets!')

    if h5_spec_inds is not None and h5_spec_vals is not None:
        # The provided datasets override fresh building instructions.
        validate_anc_h5_dsets(h5_spec_inds, h5_spec_vals, main_shape, is_spectroscopic=True)
        if verbose:
            print('The shapes of the provided h5 position indices and values '
                  'are OK')
        h5_spec_inds, h5_spec_vals = __ensure_anc_in_correct_file(h5_spec_inds, h5_spec_vals,
                                         'Spectroscopic')
    else:
        aux_spec_prefix = __check_anc_before_creation(aux_spec_prefix, dim_type='spec')
        spec_dims = validate_dimensions(spec_dims, dim_type='Spectroscopic')
        validate_dims_against_main(main_shape, spec_dims, is_spectroscopic=True)
        if verbose:
            print('Passed all pre-tests for creating spectroscopic datasets')
        h5_spec_inds, h5_spec_vals = write_ind_val_dsets(h5_parent_group, spec_dims, is_spectral=True, verbose=verbose,
                                                         slow_to_fast=slow_to_fast, base_name=aux_spec_prefix)
        if verbose:
            print('Created Spectroscopic datasets')

    if h5_parent_group.file.driver == 'mpio':
        if kwargs.pop('compression', None) is not None:
            warn('This HDF5 file has been opened wth the "mpio" communicator. '
                 'mpi4py does not allow creation of compressed datasets. Compression kwarg has been removed')

    if isinstance(main_data, np.ndarray):
        # Case 1 - simple small dataset
        h5_main = h5_parent_group.create_dataset(main_data_name, data=main_data, **kwargs)
        if verbose:
            print('Created main dataset with provided data')
    elif isinstance(main_data, da.core.Array):
        # Case 2 - Dask dataset
        # step 0 - get rid of any automated dtype specification:
        _ = kwargs.pop('dtype', None)
        # step 1 - create the empty dataset:
        h5_main = h5_parent_group.create_dataset(main_data_name, shape=main_data.shape, dtype=main_data.dtype,
                                                 **kwargs)
        if verbose:
            print('Created empty dataset: {} for writing Dask dataset: {}'.format(h5_main, main_data))
            print('Dask array will be written to HDF5 dataset: "{}" in file: "{}"'.format(h5_main.name,
                                                                                          h5_main.file.filename))
        # Step 2 - now ask Dask to dump data to disk
        da.to_hdf5(h5_main.file.filename, {h5_main.name: main_data})
        # main_data.to_hdf5(h5_main.file.filename, h5_main.name)  # Does not work with python 2 for some reason
    else:
        # Case 3 - large empty dataset
        h5_main = h5_parent_group.create_dataset(main_data_name, main_data, **kwargs)
        if verbose:
            print('Created empty dataset for Main')

    write_simple_attrs(h5_main, {'quantity': quantity, 'units': units})
    if verbose:
        print('Wrote quantity and units attributes to main dataset')

    if isinstance(main_dset_attrs, dict):
        write_simple_attrs(h5_main, main_dset_attrs)
        if verbose:
            print('Wrote provided attributes to main dataset')

    write_book_keeping_attrs(h5_main)

    # make it main
    link_as_main(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)
    if verbose:
        print('Successfully linked datasets - dataset should be main now')

    from ..usi_data import USIDataset
    return USIDataset(h5_main)


def map_grid_to_cartesian(h5_main, grid_shape, mode='histogram', **kwargs):
    """
    Map an incomplete measurement, such as a spiral scan, to a cartesian grid.
    Parameters
    ----------
    h5_main : :class:`pyUSID.USIDataset`
        Dataset containing the sparse measurement
    grid_shape : int or [int, int]
        Shape of the output :class:`numpy.ndarray`.
    mode : str, optional. Default = 'histogram'
        Method used for building a cartesian grid.
        Available methods = 'histogram', 'linear', 'nearest', 'cubic'
        Use kwargs to pass onto each of the techniques
    Note
    ----
    UNDER DEVELOPMENT!
    Currently only valid for 2 position dimensions

    @author: Patrik Marschalik

    Returns
    -------
    :class:`numpy.ndarray` but could be a h5py.Dataset or dask.array.core.Array object
    """
    try:
        from scipy.interpolate import griddata
    except ImportError as expn:
        griddata = None
        warn('map_grid_to_cartesian() requires scipy')
        raise expn

    from ..usi_data import USIDataset

    if not isinstance(h5_main, USIDataset):
        raise TypeError('Provided object is not a pyUSID.USIDataset object')

    if mode not in ['histogram', 'linear', 'nearest', 'cubic']:
        raise ValueError('mode must be a string among["histogram", "cubic"]')

    ds_main = h5_main[()].squeeze()
    ds_pos_vals = h5_main.h5_pos_vals[()]

    if ds_pos_vals.shape[1] != 2:
        raise TypeError("Only working for 2 position dimensions.")

    # Transform to row, col image format
    rotation = np.array([[0, 1], [-1, 0]])
    ds_pos_vals = np.dot(ds_pos_vals, rotation)

    try:
        grid_n = len(grid_shape)
    except TypeError:
        grid_n = 1

    if grid_n != 1 and grid_n != 2:
        raise ValueError("grid_shape must be of type int or [int, int].")

    if grid_n == 1:
        grid_shape = 2 * [grid_shape]

    def interpolate(points, values, grid_shape, method):
        grid_shape = list(map((1j).__mul__, grid_shape))
        grid_x, grid_y = np.mgrid[
            np.amin(points[:, 0]):np.amax(points[:, 0]):grid_shape[0],
            np.amin(points[:, 1]):np.amax(points[:, 1]):grid_shape[1]
        ]
        ndim_data = griddata(points, values, (grid_x, grid_y), method=method)

        return ndim_data

    if mode == "histogram":
        histogram_weighted, _, _ = np.histogram2d(*ds_pos_vals.T, bins=grid_shape, weights=ds_main)
        histogram, _, _ = np.histogram2d(*ds_pos_vals.T, bins=grid_shape)
        cart_data = np.divide(histogram_weighted, histogram)
    else:
        cart_data = interpolate(ds_pos_vals, ds_main, grid_shape, method=mode)

    return cart_data
