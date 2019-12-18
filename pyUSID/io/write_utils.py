# -*- coding: utf-8 -*-
"""
Utilities that assist in writing USID related data to HDF5 files

Created on Thu Sep  7 21:14:25 2017

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import sys
from warnings import warn
from enum import Enum
from itertools import groupby
import numpy as np
from .dtype_utils import contains_integers, validate_list_of_strings, validate_single_string_arg
if sys.version_info.major == 3:
    from collections.abc import Iterable
else:
    from collections import Iterable

__all__ = ['clean_string_att', 'get_aux_dset_slicing', 'make_indices_matrix', 'INDICES_DTYPE', 'VALUES_DTYPE', 'get_slope',
           'Dimension', 'build_ind_val_matrices', 'calc_chunks', 'create_spec_inds_from_vals', 'validate_dimensions', 'DimType',
           'to_ranges']

if sys.version_info.major == 3:
    unicode = str

# Constants:
INDICES_DTYPE = np.uint32
VALUES_DTYPE = np.float32


class DimType(Enum):
    DEFAULT = 0
    INCOMPLETE = 1
    DEPENDENT = 2

    @staticmethod
    def __check_other_type(other):
        if not isinstance(other, DimType):
            raise TypeError('Provided object not of type DimType')

    def __lt__(self, other):
        self.__check_other_type(other)
        return self.value < other.value

    def __gt__(self, other):
        self.__check_other_type(other)
        return self.value > other.value

    def __eq__(self, other):
        self.__check_other_type(other)
        return self.value == other.value


class Dimension(object):
    """
    ..autoclass::Dimension
    """

    def __init__(self, name, units, values, mode=DimType.DEFAULT):
        """
        Simple object that describes a dimension in a dataset by its name, units, and values

        Parameters
        ----------
        name : str or unicode
            Name of the dimension. For example 'Bias'
        units : str or unicode
            Units for this dimension. For example: 'V'
        values : array-like or int
            Values over which this dimension was varied. A linearly increasing set of values will be generated if an
            integer is provided instead of an array.
        mode : Enum, Optional. Default = DimType.DEFAULT
            How the parameter associated with the dimension was varied.
            DimType.DEFAULT - data was recorded for all combinations of values in this dimension against **all** other
            dimensions. This is typically the case.
            DimType.INCOMPLETE - Data not present for all combinations of values in this dimension and all other
                dimensions. Examples include spiral scans, sparse sampling, aborted measurements
            DimType.DEPENDENT - Values in this dimension were varied as a function of another (independent) dimension.
        """
        name = validate_single_string_arg(name, 'name')

        if not isinstance(units, (str, unicode)):
            raise TypeError('units should be a string')
        units = units.strip()

        if isinstance(values, int):
            if values < 1:
                raise ValueError('values should at least be specified as a positive integer')
            values = np.arange(values)
        if not isinstance(values, (np.ndarray, list, tuple)):
            raise TypeError('values should be array-like')

        if not isinstance(mode, DimType):
            raise TypeError('mode must be of type pyUSID.DimType. Provided object was of type: {}'.format(type(mode)))

        self.name = name
        self.units = units
        self.values = values
        self.mode = mode

    def __repr__(self):
        return '{} ({}) mode:{} : {}'.format(self.name, self.units, self.mode, self.values)

    def __eq__(self, other):
        if isinstance(other, Dimension):
            if self.name != other.name:
                return False
            if self.units != other.units:
                return False
            if self.mode != other.mode:
                return False
            if len(self.values) != len(other.values):
                return False
            if not np.allclose(self.values, other.values):
                return False

        return True


def validate_dimensions(dimensions, dim_type='Position'):
    """
    Checks if the provided object is an iterable with pyUSID.Dimension objects.
    If it is not full of Dimension objects, Exceptions are raised.

    Parameters
    ----------
    dimensions : iterable or pyUSID.Dimension
        Iterable containing pyUSID.Dimension objects
    dim_type : str, Optional. Default = "Position"
        Type of Dimensions in the iterable. Set to "Spectroscopic" if not Position dimensions.
        This string is only used for more descriptive Exceptions

    Returns
    -------
    list
        List containing pyUSID.Dimension objects
    """
    if isinstance(dimensions, Dimension):
        dimensions = [dimensions]
    if isinstance(dimensions, np.ndarray):
        if dimensions.ndim > 1:
            dimensions = dimensions.ravel()
            warn(dim_type + ' dimensions should be specified by a 1D array-like. Raveled this numpy array for now')
    if not isinstance(dimensions, (list, np.ndarray, tuple)):
        raise TypeError(dim_type + ' dimensions should be array-like of Dimension objects')
    if not np.all([isinstance(x, Dimension) for x in dimensions]):
        raise TypeError(dim_type + ' dimensions should be a sequence of Dimension objects')
    return dimensions


def get_aux_dset_slicing(dim_names, last_ind=None, is_spectroscopic=False):
    """
    Returns a dictionary of slice objects to help in creating region references in the position or spectroscopic
    indices and values datasets

    Parameters
    ------------
    dim_names : iterable
        List of strings denoting the names of the position axes or spectroscopic dimensions arranged in the same order
        that matches the dimensions in the indices / values dataset
    last_ind : (Optional) unsigned int, default = None
        Last pixel in the positon or spectroscopic matrix. Useful in experiments where the
        parameters have changed (eg. BEPS new data format) during the experiment.
    is_spectroscopic : bool, optional. default = True
        set to True for position datasets and False for spectroscopic datasets
    Returns
    ------------
    slice_dict : dictionary
        Dictionary of tuples containing slice objects corresponding to
        each position axis.
    """
    dim_names = validate_list_of_strings(dim_names, 'dim_names')
    if len(dim_names) == 0:
        raise ValueError('No valid dim_names provided')

    slice_dict = dict()
    for spat_ind, curr_dim_name in enumerate(dim_names):
        val = (slice(last_ind), slice(spat_ind, spat_ind + 1))
        if is_spectroscopic:
            val = val[::-1]
        slice_dict[str(curr_dim_name)] = val
    return slice_dict


def make_indices_matrix(num_steps, is_position=True):
    """
    Makes an ancillary indices matrix given the number of steps in each dimension. In other words, this function builds
    a matrix whose rows correspond to unique combinations of the multiple dimensions provided.

    Parameters
    ------------
    num_steps : List / numpy array
        Number of steps in each spatial or spectral dimension
        Note that the axes must be ordered from fastest varying to slowest varying
    is_position : bool, optional, default = True
        Whether the returned matrix is meant for position (True) indices (tall and skinny) or spectroscopic (False)
        indices (short and wide)

    Returns
    --------------
    indices_matrix : 2D unsigned int numpy array
        arranged as [steps, spatial dimension]
    """
    if not isinstance(num_steps, (tuple, list, np.ndarray)):
        raise TypeError('num_steps should be a list / tuple / numpy array')
    if not contains_integers(num_steps, min_val=1 + int(len(num_steps) > 0)):
        raise ValueError('num_steps should contain integers greater than equal to 1 (empty dimension) or 2')

    num_steps = np.array(num_steps)
    spat_dims = max(1, len(np.where(num_steps > 1)[0]))

    indices_matrix = np.zeros(shape=(np.prod(num_steps), spat_dims), dtype=INDICES_DTYPE)
    dim_ind = 0

    for indx, curr_steps in enumerate(num_steps):
        if curr_steps > 1:

            part1 = np.prod(num_steps[:indx + 1])

            if indx > 0:
                part2 = np.prod(num_steps[:indx])
            else:
                part2 = 1

            if indx + 1 == len(num_steps):
                part3 = 1
            else:
                part3 = np.prod(num_steps[indx + 1:])

            indices_matrix[:, dim_ind] = np.tile(np.floor(np.arange(part1) / part2), part3)
            dim_ind += 1

    if not is_position:
        indices_matrix = indices_matrix.T

    return indices_matrix


def clean_string_att(att_val):
    """
    Replaces any unicode objects within lists with their string counterparts to ensure compatibility with python 3.
    If the attribute is indeed a list of unicodes, the changes will be made in-place

    Parameters
    ----------
    att_val : object
        Attribute object

    Returns
    -------
    att_val : object
        Attribute object
    """
    try:
        if isinstance(att_val, Iterable):
            if type(att_val) in [unicode, str]:
                return att_val
            elif np.any([type(x) in [str, unicode, bytes, np.str_] for x in att_val]):
                return np.array(att_val, dtype='S')
        if type(att_val) == np.str_:
            return str(att_val)
        return att_val
    except TypeError:
        raise TypeError('Failed to clean: {}'.format(att_val))


def build_ind_val_matrices(unit_values, is_spectral=True):
    """
    Builds indices and values matrices using given unit values for each dimension.
    Unit values must be arranged from fastest varying to slowest varying

    Parameters
    ----------
    unit_values : list / tuple
        Sequence of values vectors for each dimension
    is_spectral : bool (optional), default = True
        If true, returns matrices for spectroscopic datasets, else returns matrices for Position datasets

    Returns
    -------
    ind_mat : 2D numpy array
        Indices matrix
    val_mat : 2D numpy array
        Values matrix
    """
    if not isinstance(unit_values, (list, tuple)):
        raise TypeError('unit_values should be a list or tuple')
    if not np.all([np.array(x).ndim == 1 for x in unit_values]):
        raise ValueError('unit_values should only contain 1D array')
    lengths = [len(x) for x in unit_values]
    tile_size = [np.prod(lengths[x:]) for x in range(1, len(lengths))] + [1]
    rep_size = [1] + [np.prod(lengths[:x]) for x in range(1, len(lengths))]
    val_mat = np.zeros(shape=(len(lengths), np.prod(lengths)))
    ind_mat = np.zeros(shape=val_mat.shape, dtype=np.uint32)
    for ind, ts, rs, vec in zip(range(len(lengths)), tile_size, rep_size, unit_values):
        val_mat[ind] = np.tile(np.repeat(vec, rs), ts)
        ind_mat[ind] = np.tile(np.repeat(np.arange(len(vec)), rs), ts)
    if not is_spectral:
        val_mat = val_mat.T
        ind_mat = ind_mat.T
    return INDICES_DTYPE(ind_mat), VALUES_DTYPE(val_mat)


def create_spec_inds_from_vals(ds_spec_val_mat):
    """
    Create new Spectroscopic Indices table from the changes in the
    Spectroscopic Values

    Parameters
    ----------
    ds_spec_val_mat : array-like,
        Holds the spectroscopic values to be indexed

    Returns
    -------
    ds_spec_inds_mat : numpy array of uints the same shape as ds_spec_val_mat
        Indices corresponding to the values in ds_spec_val_mat

    """
    if not isinstance(ds_spec_val_mat, np.ndarray):
        raise TypeError('ds_spec_val_mat must be a numpy array')
    if ds_spec_val_mat.ndim != 2:
        raise ValueError('ds_spec_val_mat must be a 2D array arranged as [dimension, values]')

    ds_spec_inds_mat = np.zeros_like(ds_spec_val_mat, dtype=np.int32)

    """
    Find how quickly the spectroscopic values are changing in each row 
    and the order of row from fastest changing to slowest.
    """
    change_count = [len(np.where([row[i] != row[i - 1] for i in range(len(row))])[0]) for row in ds_spec_val_mat]
    change_sort = np.argsort(change_count)[::-1]

    """
    Determine everywhere the spectroscopic values change and build 
    index table based on those changed
    """
    indices = np.zeros(ds_spec_val_mat.shape[0])
    for jcol in range(1, ds_spec_val_mat.shape[1]):
        this_col = ds_spec_val_mat[change_sort, jcol]
        last_col = ds_spec_val_mat[change_sort, jcol - 1]

        """
        Check if current column values are different than those 
        in last column.
        """
        changed = np.where(this_col != last_col)[0]

        """
        If only one row changed, increment the index for that 
        column
        If more than one row has changed, increment the index for 
        the last row that changed and set all others to zero
        """
        if len(changed) == 1:
            indices[changed] += 1
        elif len(changed > 1):
            for change in changed[:-1]:
                indices[change] = 0
            indices[changed[-1]] += 1

        """
        Store the indices for the current column in the dataset
        """
        ds_spec_inds_mat[change_sort, jcol] = indices

    return ds_spec_inds_mat


def calc_chunks(dimensions, dtype_byte_size, unit_chunks=None, max_chunk_mem=10240):
    """
    Calculate the chunk size for the HDF5 dataset based on the dimensions and the
    maximum chunk size in memory

    Parameters
    ----------
    dimensions : array_like of int
        Shape of the data to be chunked
    dtype_byte_size : unsigned int
        Size of an entry in the data in bytes
    unit_chunks : array_like of int, optional
        Unit size of the chunking in each dimension.  Must be the same size as
        the shape of `ds_main`.  Default None, `unit_chunks` is set to 1 in all
        dimensions
    max_chunk_mem : int, optional
        Maximum size of the chunk in memory in bytes.  Default 10240b or 10kb per h5py recommendations

    Returns
    -------
    chunking : tuple of int
        Calculated maximum size of a chunk in each dimension that is as close to the
        requested `max_chunk_mem` as posible while having steps based on the input
        `unit_chunks`.
    """
    if not isinstance(dimensions, (list, tuple)):
        raise TypeError('dimensions should either be a tuple or list')
    if not isinstance(dtype_byte_size, int):
        raise TypeError('dtype_byte_size should be an integer')
    if unit_chunks is not None:
        if not isinstance(unit_chunks, (tuple, list)):
            raise TypeError('unit_chunks should either be a tuple or list')

    '''
    Ensure that dimensions is an array
    '''
    dimensions = np.asarray(dimensions, dtype=np.uint)
    '''
    Set the unit_chunks to all ones if not given.  Ensure it is an array if it is.
    '''
    if unit_chunks is None:
        unit_chunks = np.ones_like(dimensions)
    else:
        unit_chunks = np.asarray(unit_chunks, dtype=np.uint)

    if unit_chunks.shape != dimensions.shape:
        raise ValueError('Unit chunk size must have the same shape as the input dataset.')

    '''
    Save the original size of unit_chunks to use for incrementing the chunk size during
     loop
    '''
    base_chunks = unit_chunks.copy()

    '''
    Loop until chunk_size is greater than the maximum chunk_mem or the chunk_size is equal to
    that of dimensions
    '''
    while np.prod(unit_chunks) * dtype_byte_size <= max_chunk_mem:
        '''
        Check if all chunk dimensions are greater or equal to the
        actual dimensions.  Exit the loop if true.
        '''
        if np.all(unit_chunks >= dimensions):
            break

        '''
        Find the index of the next chunk to be increased and increment it by the base_chunk
        size
        '''
        ichunk = np.argmax(dimensions / unit_chunks)
        unit_chunks[ichunk] += base_chunks[ichunk]

    '''
    Ensure that the size of the chunks is between one and the dimension size.
    '''
    unit_chunks = np.clip(unit_chunks, np.ones_like(unit_chunks), dimensions)

    chunking = tuple(unit_chunks)

    return chunking


def get_slope(values, tol=1E-3):
    """
    Attempts to get the slope of the provided values. This function will be handy
    for checking if a dimension has been varied linearly or not.
    If the values vary non-linearly, a ValueError will be raised

    Parameters
    ----------
    values : array-like
        List of numbers
    tol : float, optional. Default = 1E-3
        Tolerance in the variation of the slopes.
    Returns
    -------
    float
        Slope of the line
    """
    if not isinstance(tol, float):
        raise TypeError('tol should be a float << 1')
    step_size = np.unique(np.diff(values))
    if len(step_size) > 1:
        # often we end up here. In most cases,
        step_avg = step_size.max()
        step_size -= step_avg
        var = np.mean(np.abs(step_size))
        if var / step_avg < tol:
            step_size = [step_avg]
        else:
            # Non-linear dimension! - see notes above
            raise ValueError('Provided values cannot be expressed as a linear trend')
    return step_size[0]


def to_ranges(iterable):
    """
    Converts a sequence of iterables to range tuples

    From https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python

    Credits: @juanchopanza and @luca

    Parameters
    ----------
    iterable : collections.Iterable object
        iterable object like a list

    Returns
    -------
    iterable : generator object
        Cast to list or similar to use
    """
    iterable = sorted(set(iterable))
    for key, group in groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        if sys.version_info.major == 3:
            yield range(group[0][1], group[-1][1]+1)
        else:
            yield xrange(group[0][1], group[-1][1]+1)

