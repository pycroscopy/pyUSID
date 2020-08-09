# -*- coding: utf-8 -*-
"""
Utilities for transforming and validating data types

Given that many of the data transformations involve copying the data, they should
ideally happen in a lazy manner to avoid memory issues.

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, absolute_import, unicode_literals, print_function
from warnings import warn
import sidpy

__all__ = ['flatten_complex_to_real', 'get_compound_sub_dtypes', 'flatten_compound_to_real', 'check_dtype',
           'stack_real_to_complex', 'validate_dtype', 'integers_to_slices', 'get_exponent', 'is_complex_dtype',
           'stack_real_to_compound', 'stack_real_to_target_dtype', 'flatten_to_real', 'contains_integers',
           'validate_single_string_arg', 'validate_string_args', 'validate_list_of_strings',
           'lazy_load_array']


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
    warn('pyUSID.io.dtype_utils.lazy_load_array has been moved to '
         'sidpy.hdf.hdf_utils.lazy_load_array. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.hdf_utils.lazy_load_array(dataset)


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
    warn('pyUSID.io.dtype_utils.contains_integers has been moved to '
         'sidpy.base.num_utils.contains_integers. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.base.num_utils.contains_integers(iter_int, min_val=min_val)


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
    warn('pyUSID.io.dtype_utils.flatten_complex_to_real has been moved to '
         'sidpy.hdf.dtype_utils.flatten_complex_to_real. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.flatten_complex_to_real(dataset, lazy=lazy)


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
    warn('pyUSID.io.dtype_utils.flatten_compound_to_real has been moved to '
         'sidpy.hdf.dtype_utils.flatten_compound_to_real. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.flatten_compound_to_real(dataset, lazy=lazy)


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
    warn('pyUSID.io.dtype_utils.flatten_to_real has been moved to '
         'sidpy.hdf.dtype_utils.flatten_to_real. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.flatten_to_real(ds_main, lazy=lazy)


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
    warn('pyUSID.io.dtype_utils.get_compound_sub_dtypes has been moved to '
         'sidpy.hdf.dtype_utils.get_compound_sub_dtypes. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.get_compound_sub_dtypes(struct_dtype)


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
    warn('pyUSID.io.dtype_utils.check_dtype has been moved to '
         'sidpy.hdf.dtype_utils.check_dtype. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.check_dtype(h5_dset)


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
    warn('pyUSID.io.dtype_utils.stack_real_to_complex has been moved to '
         'sidpy.hdf.dtype_utils.stack_real_to_complex. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.stack_real_to_complex(ds_real, lazy=lazy)


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
    warn('pyUSID.io.dtype_utils.stack_real_to_compound has been moved to '
         'sidpy.hdf.dtype_utils.stack_real_to_compound. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.stack_real_to_compound(ds_real, compound_type,
                                                        lazy=lazy)


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
    warn('pyUSID.io.dtype_utils.stack_real_to_target_dtype has been moved to '
         'sidpy.hdf.dtype_utils.stack_real_to_target_dtype. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.stack_real_to_target_dtype(ds_real, new_dtype,
                                                            lazy=lazy)


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
    warn('pyUSID.io.dtype_utils.validate_dtype has been moved to '
         'sidpy.hdf.dtype_utils.validate_dtype. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.validate_dtype(dtype)


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
    warn('pyUSID.io.dtype_utils.validate_single_string_arg has been moved to '
         'sidpy.base.string_utils.validate_single_string_arg. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.base.string_utils.validate_single_string_arg(value, name)


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

    warn('pyUSID.io.dtype_utils.validate_list_of_strings has been moved to '
         'sidpy.base.string_utils.validate_list_of_strings. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.base.string_utils.validate_list_of_strings(str_list, parm_name=parm_name)


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
    warn('pyUSID.io.dtype_utils.validate_string_args has been moved to '
         'sidpy.base.string_utils.validate_string_args. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.base.string_utils.validate_string_args(arg_list, arg_names)


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
    warn('pyUSID.io.dtype_utils.is_complex_dtype has been moved to '
         'sidpy.hdf.dtype_utils.is_complex_dtype. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.hdf.dtype_utils.is_complex_dtype(dtype)


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
    warn('pyUSID.io.dtype_utils.integers_to_slices has been moved to '
         'sidpy.base.num_utils.integers_to_slices. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.base.num_utils.integers_to_slices(int_array)


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
    warn('pyUSID.io.dtype_utils.get_exponent has been moved to '
         'sidpy.base.num_utils.get_exponent. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sidpy.base.num_utils.get_exponent(vector)
