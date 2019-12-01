# -*- coding: utf-8 -*-
"""
Utilities for transforming and validating data types

Given that many of the data transformations involve copying the data, they should
ideally happen in a lazy manner to avoid memory issues.

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, absolute_import, unicode_literals, print_function
import sys
from warnings import warn
import h5py
import numpy as np
import dask.array as da
from itertools import groupby
if sys.version_info.major == 3:
    from collections.abc import Iterable
else:
    from collections import Iterable

__all__ = ['flatten_complex_to_real', 'get_compound_sub_dtypes', 'flatten_compound_to_real', 'check_dtype',
           'stack_real_to_complex', 'validate_dtype', 'integers_to_slices', 'get_exponent', 'is_complex_dtype',
           'stack_real_to_compound', 'stack_real_to_target_dtype', 'flatten_to_real', 'contains_integers',
           'validate_single_string_arg', 'validate_string_args', 'validate_list_of_strings',
           'lazy_load_array']

if sys.version_info.major == 3:
    unicode = str


def lazy_load_array(dataset):
    """
    Loads the provided object as a dask array (h5py.Dataset or numpy.ndarray

    Parameters
    ----------
    dataset : :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or :class:`dask.array.core.Array`
        Array to laod as dask array

    Returns
    -------
    :class:`dask.array.core.Array`
        Dask array with appropriate chunks
    """
    if isinstance(dataset, da.core.Array):
        return dataset
    elif not isinstance(dataset, (h5py.Dataset, np.ndarray)):
        raise TypeError('Expected one of h5py.Dataset, dask.array.core.Array, or numpy.ndarray'
                        'objects. Provided object was of type: {}'.format(type(dataset)))
    # Cannot pass 'auto' for chunks for python 2!
    chunks = "auto" if sys.version_info.major == 3 else dataset.shape
    if isinstance(dataset, h5py.Dataset):
        chunks = chunks if dataset.chunks is None else dataset.chunks
    return da.from_array(dataset, chunks=chunks)


def contains_integers(iter_int, min_val=None):
    """
    Checks if the provided object is iterable (list, tuple etc.) and contains integers optionally greater than equal to
    the provided min_val

    Parameters
    ----------
    iter_int : :class:`collections.Iterable`
        Iterable (e.g. list, tuple, etc.) of integers
    min_val : int, optional, default = None
        The value above which each element of iterable must possess. By default, this is ignored.

    Returns
    -------
    bool
        Whether or not the provided object is an iterable of integers
    """
    if not isinstance(iter_int, Iterable):
        raise TypeError('iter_int should be an Iterable')
    if len(iter_int) == 0:
        return False

    if min_val is not None:
        if not isinstance(min_val, (int, float)):
            raise TypeError('min_val should be an integer. Provided object was of type: {}'.format(type(min_val)))
        if min_val % 1 != 0:
            raise ValueError('min_val should be an integer')

    try:
        if min_val is not None:
            return np.all([x % 1 == 0 and x >= min_val for x in iter_int])
        else:
            return np.all([x % 1 == 0 for x in iter_int])
    except TypeError:
        return False


def flatten_complex_to_real(dataset, lazy=False):
    """
    Stacks the real values followed by the imaginary values in the last dimension of the given N dimensional matrix.
    Thus a complex matrix of shape (2, 3, 5) will turn into a matrix of shape (2, 3, 10)

    Parameters
    ----------
    dataset : array-like or :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or :class:`dask.array.core.Array`
        Dataset of complex data type
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    -------
    retval : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
        real valued dataset
    """
    if not isinstance(dataset, (h5py.Dataset, np.ndarray, da.core.Array)):
        raise TypeError('dataset should either be a h5py.Dataset or numpy / dask array')
    if not is_complex_dtype(dataset.dtype):
        raise TypeError("Expected a complex valued dataset")

    if isinstance(dataset, da.core.Array):
        lazy = True

    xp = np
    if lazy:
        dataset = lazy_load_array(dataset)
        xp = da

    axis = xp.array(dataset).ndim - 1
    if axis == -1:
        return xp.hstack([xp.real(dataset), xp.imag(dataset)])
    else:  # along the last axis
        return xp.concatenate([xp.real(dataset), xp.imag(dataset)], axis=axis)


def flatten_compound_to_real(dataset, lazy=False):
    """
    Flattens the individual components in a structured array or compound valued hdf5 dataset along the last axis to form
    a real valued array. Thus a compound h5py.Dataset or structured numpy matrix of shape (2, 3, 5) having 3 components
    will turn into a real valued matrix of shape (2, 3, 15), assuming that all the sub-dtypes of the matrix are real
    valued. ie - this function does not handle structured dtypes having complex values


    Parameters
    ----------
    dataset : :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or :class:`dask.array.core.Array`
        Numpy array that is a structured array or a :class:`h5py.Dataset` of compound dtype
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    -------
    retval : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
        real valued dataset
    """
    if isinstance(dataset, h5py.Dataset):
        if len(dataset.dtype) == 0:
            raise TypeError("Expected compound h5py dataset")

        if lazy:
            xp = da
            dataset = lazy_load_array(dataset)
        else:
            xp = np
            warn('HDF5 datasets will be loaded as Dask arrays in the future. ie - kwarg lazy will default to True in future releases of pyUSID')

        return xp.concatenate([xp.array(dataset[name]) for name in dataset.dtype.names], axis=len(dataset.shape) - 1)

    elif isinstance(dataset, (np.ndarray, da.core.Array)):
        if isinstance(dataset, da.core.Array):
            lazy = True

        xp = np
        if lazy:
            dataset = lazy_load_array(dataset)
            xp = da

        if len(dataset.dtype) == 0:
            raise TypeError("Expected structured array")
        if dataset.ndim > 0:
            return xp.concatenate([dataset[name] for name in dataset.dtype.names], axis=dataset.ndim - 1)
        else:
            return xp.hstack([dataset[name] for name in dataset.dtype.names])
    elif isinstance(dataset, np.void):
        return np.hstack([dataset[name] for name in dataset.dtype.names])
    else:
        raise TypeError('Datatype {} not supported'.format(type(dataset)))


def flatten_to_real(ds_main, lazy=False):
    """
    Flattens complex / compound / real valued arrays to real valued arrays

    Parameters
    ----------
    ds_main : :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or :class:`dask.array.core.Array`
        Compound, complex or real valued numpy array or HDF5 dataset
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    ----------
    ds_main : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
        Array raveled to a float data type
    """
    if not isinstance(ds_main, (h5py.Dataset, np.ndarray, da.core.Array)):
        ds_main = np.array(ds_main)
    if is_complex_dtype(ds_main.dtype):
        return flatten_complex_to_real(ds_main, lazy=lazy)
    elif len(ds_main.dtype) > 0:
        return flatten_compound_to_real(ds_main, lazy=lazy)
    else:
        return ds_main


def get_compound_sub_dtypes(struct_dtype):
    """
    Returns a dictionary of the dtypes of each of the fields in the given structured array dtype

    Parameters
    ----------
    struct_dtype : :class:`numpy.dtype`
        dtype of a structured array

    Returns
    -------
    dtypes : dict
        Dictionary whose keys are the field names and values are the corresponding dtypes
    """
    if not isinstance(struct_dtype, np.dtype):
        raise TypeError('Provided object must be a structured array dtype')
    dtypes = dict()
    for field_name in struct_dtype.fields:
        dtypes[field_name] = struct_dtype.fields[field_name][0]
    return dtypes


def check_dtype(h5_dset):
    """
    Checks the datatype of the input HDF5 dataset and provides the appropriate
    function calls to convert it to a float

    Parameters
    ----------
    h5_dset : :class:`h5py.Dataset`
        Dataset of interest

    Returns
    -------
    func : callable
        function that will convert the dataset to a float
    is_complex : bool
        is the input dataset complex?
    is_compound : bool
        is the input dataset compound?
    n_features : Unsigned int
        Unsigned integer - the length of the 2nd dimension of the data after `func` is called on it
    type_mult : Unsigned int
        multiplier that converts from the typesize of the input :class:`~numpy.dtype` to the
        typesize of the data after func is run on it
    """
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object')
    is_complex = False
    is_compound = False
    in_dtype = h5_dset.dtype
    # TODO: avoid assuming 2d shape - why does one even need n_samples!? We only care about the last dimension!
    n_features = h5_dset.shape[-1]
    if is_complex_dtype(h5_dset.dtype):
        is_complex = True
        new_dtype = np.real(h5_dset[0, 0]).dtype
        type_mult = new_dtype.itemsize * 2
        func = flatten_complex_to_real
        n_features *= 2
    elif len(h5_dset.dtype) > 0:
        """
        Some form of structured numpy is in use
        We only support real scalars for the component types at the current time
        """
        is_compound = True
        # TODO: Avoid hard-coding to float32
        new_dtype = np.float32
        type_mult = len(in_dtype) * new_dtype(0).itemsize
        func = flatten_compound_to_real
        n_features *= len(in_dtype)
    else:
        if h5_dset.dtype not in [np.float32, np.float64]:
            new_dtype = np.float32
        else:
            new_dtype = h5_dset.dtype.type

        type_mult = new_dtype(0).itemsize

        func = new_dtype

    return func, is_complex, is_compound, n_features, type_mult


def stack_real_to_complex(ds_real, lazy=False):
    """
    Puts the real and imaginary sections of the provided matrix (in the last axis) together to make complex matrix

    Parameters
    ------------
    ds_real : :class:`numpy.ndarray`, :class:`dask.array.core.Array`, or :class:`h5py.Dataset`
        n dimensional real-valued numpy array or HDF5 dataset where data arranged as [instance, 2 x features],
        where the first half of the features are the real component and the
        second half contains the imaginary components
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    ----------
    ds_compound : :class:`numpy.ndarray` or :class:`dask.array.core.Array`
        2D complex array arranged as [sample, features]
    """
    if not isinstance(ds_real, (np.ndarray, da.core.Array, h5py.Dataset)):
        if not isinstance(ds_real, (tuple, list)):
            raise TypeError("Expected at least an iterable like a list or tuple")
        ds_real = np.array(ds_real)
    if len(ds_real.dtype) > 0:
        raise TypeError("Array cannot have a compound dtype")
    if is_complex_dtype(ds_real.dtype):
        raise TypeError("Array cannot have complex dtype")

    if ds_real.shape[-1] / 2 != ds_real.shape[-1] // 2:
        raise ValueError("Last dimension must be even sized")
    half_point = ds_real.shape[-1] // 2

    if isinstance(ds_real, da.core.Array):
        lazy = True

    if lazy and not isinstance(ds_real, da.core.Array):
        ds_real = lazy_load_array(ds_real)

    return ds_real[..., :half_point] + 1j * ds_real[..., half_point:]


def stack_real_to_compound(ds_real, compound_type, lazy=False):
    """
    Converts a real-valued dataset to a compound dataset (along the last axis) of the provided compound d-type

    Parameters
    ------------
    ds_real : :class:`numpy.ndarray`, :class:`dask.array.core.Array`, or :class:`h5py.Dataset`
        n dimensional real-valued numpy array or HDF5 dataset where data arranged as [instance, features]
    compound_type : :class:`numpy.dtype`
        Target complex data-type
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    ----------
    ds_compound : :class:`numpy.ndarray` or :class:`dask.array.core.Array`
        N-dimensional complex-valued array arranged as [sample, features]
    """
    if lazy or isinstance(ds_real, da.core.Array):
        raise NotImplementedError('Lazy operation not available due to absence of Dask support')
    if not isinstance(ds_real, (np.ndarray, h5py.Dataset)):
        if not isinstance(ds_real, (list, tuple)):
            raise TypeError("Expected at least an iterable like a list or tuple")
        ds_real = np.array(ds_real)
    if len(ds_real.dtype) > 0:
        raise TypeError("Array cannot have a compound dtype")
    elif is_complex_dtype(ds_real.dtype):
        raise TypeError("Array cannot have complex dtype")
    if not isinstance(compound_type, np.dtype):
        raise TypeError('Provided object must be a structured array dtype')

    new_spec_length = ds_real.shape[-1] / len(compound_type)
    if new_spec_length % 1:
        raise ValueError('Provided compound type was not compatible by number of elements')

    new_spec_length = int(new_spec_length)
    new_shape = list(ds_real.shape)  # Make mutable
    new_shape[-1] = new_spec_length

    xp = np
    kwargs = {}
    """
    if isinstance(ds_real, h5py.Dataset) and not lazy:
        warn('HDF5 datasets will be loaded as Dask arrays in the future. ie - kwarg lazy will default to True in future releases of pyUSID')
    if isinstance(ds_real, da.core.Array):
        lazy = True    
    if lazy:
        xp = da
        ds_real = lazy_load_array(ds_real)
        kwargs = {'chunks': 'auto'}
    """

    ds_compound = xp.empty(new_shape, dtype=compound_type, **kwargs)
    for name_ind, name in enumerate(compound_type.names):
        i_start = name_ind * new_spec_length
        i_end = (name_ind + 1) * new_spec_length
        ds_compound[name] = ds_real[..., i_start:i_end]

    return ds_compound.squeeze()


def stack_real_to_target_dtype(ds_real, new_dtype, lazy=False):
    """
    Transforms real data into the target dtype

    Parameters
    ----------
    ds_real : :class:`numpy.ndarray`, :class:`dask.array.core.Array` or :class:`h5py.Dataset`
        n dimensional real-valued numpy array or HDF5 dataset
    new_dtype : :class:`numpy.dtype`
        Target data-type

    Returns
    ----------
    ret_val :  :class:`numpy.ndarray` or :class:`dask.array.core.Array`
        N-dimensional array of the target data-type
    """
    if is_complex_dtype(new_dtype):
        return stack_real_to_complex(ds_real, lazy=lazy)
    try:
        if len(new_dtype) > 0:
            return stack_real_to_compound(ds_real, new_dtype, lazy=lazy)
    except TypeError:
        return new_dtype(ds_real)

    # catching all other cases, such as np.dtype('<f4')
    return new_dtype.type(ds_real)


def validate_dtype(dtype):
    """
    Checks the provided object to ensure that it is a valid dtype that can be written to an HDF5 file.
    Raises a type error if invalid. Returns True if the object passed the tests

    Parameters
    ----------
    dtype : object
        Object that is hopefully a :class:`h5py.Datatype`, or :class:`numpy.dtype` object

    Returns
    -------
    status : bool
        True if the object was a valid data-type
    """
    if isinstance(dtype, (h5py.Datatype, np.dtype)):
        pass
    elif isinstance(np.dtype(dtype), np.dtype):
        # This should catch all those instances when dtype is something familiar like - np.float32
        pass
    else:
        raise TypeError('dtype should either be a numpy or h5py dtype')
    return True


def validate_single_string_arg(value, name):
    """
    This function is to be used when validating a SINGLE string parameter for a function. Trims the provided value
    Errors in the string will result in Exceptions

    Parameters
    ----------
    value : str
        Value of the parameter
    name : str
        Name of the parameter

    Returns
    -------
    str
        Cleaned string value of the parameter
    """
    if not isinstance(value, (str, unicode)):
        raise TypeError(name + ' should be a string')
    value = value.strip()
    if len(value) <= 0:
        raise ValueError(name + ' should not be an empty string')
    return value


def validate_list_of_strings(str_list, parm_name='parameter'):
    """
    This function is to be used when validating and cleaning a list of strings. Trims the provided strings
    Errors in the strings will result in Exceptions

    Parameters
    ----------
    str_list : array-like
        list or tuple of strings
    parm_name : str, Optional. Default = 'parameter'
        Name of the parameter corresponding to this string list that will be reported in the raised Errors

    Returns
    -------
    array-like
        List of trimmed and validated strings when ALL objects within the list are found to be valid strings
    """

    if isinstance(str_list, (str, unicode)):
        return [validate_single_string_arg(str_list, parm_name)]

    if not isinstance(str_list, (list, tuple)):
        raise TypeError(parm_name + ' should be a string or list / tuple of strings')

    return [validate_single_string_arg(x, parm_name) for x in str_list]


def validate_string_args(arg_list, arg_names):
    """
    This function is to be used when validating string parameters for a function. Trims the provided strings.
    Errors in the strings will result in Exceptions

    Parameters
    ----------
    arg_list : array-like
        List of str objects that signify the value for a position argument in a function
    arg_names : array-like
        List of str objects with the names of the corresponding parameters in the function

    Returns
    -------
    array-like
        List of str objects that signify the value for a position argument in a function with spaces on ends removed
    """
    if isinstance(arg_list, (str, unicode)):
        arg_list = [arg_list]
    if isinstance(arg_names, (str, unicode)):
        arg_names = [arg_names]
    cleaned_args = []
    if not isinstance(arg_list, (tuple, list)):
        raise TypeError('arg_list should be a tuple or a list or a string')
    if not isinstance(arg_names, (tuple, list)):
        raise TypeError('arg_names should be a tuple or a list or a string')
    for arg, arg_name in zip(arg_list, arg_names):
        cleaned_args.append(validate_single_string_arg(arg, arg_name))
    return cleaned_args


def is_complex_dtype(dtype):
    """
    Checks if the provided dtype is a complex dtype

    Parameters
    ----------
    dtype : object
        Object that is a class:`h5py.Datatype`, or :class:`numpy.dtype` object

    Returns
    -------
    is_complex : bool
        True if the dtype was a complex dtype. Else returns False
    """
    validate_dtype(dtype)
    if dtype in [np.complex, np.complex64, np.complex128]:
        return True
    return False


def integers_to_slices(int_array):
    """
    Converts a sequence of iterables to a list of slice objects denoting sequences of consecutive numbers

    Parameters
    ----------
    int_array : :class:`collections.Iterable`
        iterable object like a :class:`list` or :class:`numpy.ndarray`

    Returns
    -------
    sequences : list
        List of :class:`slice` objects each denoting sequences of consecutive numbers
    """
    if not contains_integers(int_array):
        raise ValueError('Expected a list, tuple, or numpy array of integers')

    def integers_to_consecutive_sections(integer_array):
        """
        Converts a sequence of iterables to tuples with start and stop bounds

        @author: @juanchopanza and @luca from stackoverflow

        Parameters
        ----------
        integer_array : :class:`collections.Iterable`
            iterable object like a :class:`list`

        Returns
        -------
        iterable : :class:`generator`
            Cast to list or similar to use

        Note
        ----
        From https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
        """
        integer_array = sorted(set(integer_array))
        for key, group in groupby(enumerate(integer_array),
                                  lambda t: t[1] - t[0]):
            group = list(group)
            yield group[0][1], group[-1][1]

    sequences = [slice(item[0], item[1] + 1) for item in integers_to_consecutive_sections(int_array)]
    return sequences


def get_exponent(vector):
    """
    Gets the scale / exponent for a sequence of numbers. This is particularly useful when wanting to scale a vector
    for the purposes of plotting

    Parameters
    ----------
    vector : array-like
        Array of numbers

    Returns
    -------
    exponent : int
        Scale / exponent for the given vector
    """
    if not isinstance(vector, np.ndarray):
        raise TypeError('vector should be of type numpy.ndarray. Provided object of type: {}'.format(type(vector)))
    if np.max(np.abs(vector)) == np.max(vector):
        exponent = np.log10(np.max(vector))
    else:
        # negative values
        exponent = np.log10(np.max(np.abs(vector)))
    return int(np.floor(exponent))