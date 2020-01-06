# -*- coding: utf-8 -*-
"""
:class:`~pyUSID.io.usi_data.USIDataset` class that simplifies slicing, visualization, reshaping, etc. of USID datasets

Created on Thu Sep  7 21:14:25 2017

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import os
import sys
from warnings import warn
import h5py
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

from .hdf_utils import check_if_main, get_attr, create_results_group, write_reduced_anc_dsets, link_as_main, \
    get_dimensionality, get_sort_order, get_unit_values, reshape_to_n_dims, write_main_dataset, reshape_from_n_dims, \
    copy_attributes
from .dtype_utils import flatten_to_real, contains_integers, get_exponent, is_complex_dtype, \
    validate_single_string_arg, validate_list_of_strings, lazy_load_array
from .write_utils import Dimension
from ..viz.jupyter_utils import simple_ndim_visualizer
from ..viz.plot_utils import plot_map, get_plot_grid_size

if sys.version_info.major == 3:
    unicode = str


class USIDataset(h5py.Dataset):
    """
     A class that simplifies slicing, visualization, reshaping, reduction etc. of USID datasets in HDF5 files.
    This class extends the :class:`h5py.Dataset`.
    """

    def __init__(self, h5_ref, sort_dims=False):
        """
        Parameters
        ----------
        h5_ref : :class:`h5py.Dataset`
            The dataset which is actually a USID Main dataset
        sort_dims : bool, Optional. Default=False
            If set to True - Dimensions will be sorted from slowest to fastest
            Else - Dimensions will be arranged as they appear in ancillary datasets

        Methods
        -------
        self.get_current_sorting
        self.toggle_sorting
        self.get_pos_values
        self.get_spec_values
        self.get_n_dim_form
        self.slice


        Attributes
        ----------
        self.h5_spec_vals : :class:`h5py.Dataset`
            Associated Spectroscopic Values dataset
        self.h5_spec_inds : :class:`h5py.Dataset`
            Associated Spectroscopic Indices dataset
        self.h5_pos_vals : :class:`h5py.Dataset`
            Associated Position Values dataset
        self.h5_pos_inds : :class:`h5py.Dataset`
            Associated Position Indices dataset
        self.pos_dim_labels : list of str
            The labels for the position dimensions.
        self.spec_dim_labels : list of str
            The labels for the spectroscopic dimensions.
        self.n_dim_labels : list of str
            The labels for the n-dimensional dataset.
        self.pos_dim_sizes : list of int
            A list of the sizes of each position dimension.
        self.spec_dim_sizes : list of int
            A list of the sizes of each spectroscopic dimension.
        self.n_dim_sizes : list of int
            A list of the sizes of each dimension.

        Notes
        -----
        The order of all labels and sizes attributes is determined by the current value of `sort_dims`.

        """

        if not check_if_main(h5_ref):
            raise TypeError('Supply a h5py.Dataset object that is a USID main dataset')

        super(USIDataset, self).__init__(h5_ref.id)

        # User accessible properties
        # The required Position and Spectroscopic datasets
        self.h5_spec_vals = self.file[self.attrs['Spectroscopic_Values']]
        self.h5_spec_inds = self.file[self.attrs['Spectroscopic_Indices']]
        self.h5_pos_vals = self.file[self.attrs['Position_Values']]
        self.h5_pos_inds = self.file[self.attrs['Position_Indices']]

        # The dimension labels as they appear in the ancillary datasets
        self.__orig_pos_dim_labels = get_attr(self.h5_pos_inds, 'labels')
        self.__orig_spec_dim_labels = get_attr(self.h5_spec_inds, 'labels')

        # Data descriptors
        self.data_descriptor = '{} ({})'.format(get_attr(self, 'quantity'), get_attr(self, 'units'))
        self.pos_dim_descriptors = self.__get_anc_labels(self.h5_pos_inds)
        self.spec_dim_descriptors = self.__get_anc_labels(self.h5_spec_inds)

        # The size of each dimension
        self.__orig_pos_dim_sizes = np.array(get_dimensionality(np.transpose(self.h5_pos_inds)))
        self.__orig_spec_dim_sizes = np.array(get_dimensionality(np.atleast_2d(self.h5_spec_inds)))

        # Sorted dimension order
        self.__pos_sort_order = get_sort_order(np.transpose(self.h5_pos_inds))
        self.__spec_sort_order = get_sort_order(np.atleast_2d(self.h5_spec_inds))

        # internal book-keeping / we don't want users to mess with these?
        self.__orig_n_dim_sizes = np.append(self.__orig_pos_dim_sizes, self.__orig_spec_dim_sizes)
        self.__orig_n_dim_labs = np.append(self.__orig_pos_dim_labels, self.__orig_spec_dim_labels)
        self.__n_dim_sort_order_orig_s2f = np.append(self.__pos_sort_order[::-1],
                                                     self.__spec_sort_order[::-1] + len(self.__pos_sort_order))
        self.__n_dim_sort_order_orig_f2s = np.append(self.__pos_sort_order,
                                                     self.__spec_sort_order + len(self.__pos_sort_order))

        self.__n_dim_data_orig = None
        self.__n_dim_data_s2f = None
        self.__curr_ndim_form = None
        self.__n_dim_form_avail = False

        # Should the dimensions be sorted from slowest to fastest
        self.__sort_dims = sort_dims

        # Declaring var names within init
        self.pos_dim_labels = None
        self.spec_dim_labels = None
        self.pos_dim_sizes = None
        self.spec_dim_sizes = None
        self.n_dim_labels = None
        self.n_dim_sizes = None

        self.__lazy_2d = lazy_load_array(self)

        self.__set_labels_and_sizes()

        try:
            self.__n_dim_data_orig = self.get_n_dim_form(lazy=True)
            self.__n_dim_form_avail = True
            # TODO: This line keeps failing. Fix it
            self.__n_dim_data_s2f = self.__n_dim_data_orig.transpose(self.__n_dim_sort_order_orig_s2f)
        except ValueError:
            warn('This dataset does not have an N-dimensional form')

        self.__set_n_dim_view()

    def __eq__(self, other):
        if isinstance(other, h5py.Dataset):
            return super(USIDataset, self).__eq__(other)

        return False

    def __repr__(self):
        h5_str = super(USIDataset, self).__repr__()

        pos_str = ' \n'.join(['\t{} - size: {}'.format(dim_name, str(dim_size)) for dim_name, dim_size in
                              zip(self.__orig_pos_dim_labels, self.__orig_pos_dim_sizes)])
        spec_str = ' \n'.join(['\t{} - size: {}'.format(dim_name, str(dim_size)) for dim_name, dim_size in
                               zip(self.__orig_spec_dim_labels, self.__orig_spec_dim_sizes)])

        usid_str = ' \n'.join(['located at:',
                                '\t' + self.name,
                                'Data contains:', '\t' + self.data_descriptor,
                                'Data dimensions and original shape:',
                                'Position Dimensions:',
                                pos_str,
                                'Spectroscopic Dimensions:',
                                spec_str])

        if self.dtype.fields is not None:
            usid_str = '\n'.join([usid_str,
                                  'Data Fields:', '\t' + ', '.join([field for field in self.dtype.fields])])
        else:
            usid_str = '\n'.join([usid_str,
                                   'Data Type:', '\t' + self.dtype.name])

        if sys.version_info.major == 2:
            usid_str = usid_str.encode('utf8')

        return '\n'.join([h5_str, usid_str])

    def __set_labels_and_sizes(self):
        """
        Sets the labels and sizes attributes to the correct values based on
        the value of `self.__sort_dims`
        """
        if self.__sort_dims:
            self.pos_dim_labels = self.__orig_pos_dim_labels[self.__pos_sort_order].tolist()
            self.spec_dim_labels = self.__orig_spec_dim_labels[self.__spec_sort_order].tolist()
            self.pos_dim_sizes = self.__orig_pos_dim_sizes[self.__pos_sort_order].tolist()
            self.spec_dim_sizes = self.__orig_spec_dim_sizes[self.__spec_sort_order].tolist()
            self.n_dim_labels = self.__orig_n_dim_labs[self.__n_dim_sort_order_orig_s2f].tolist()
            self.n_dim_sizes = self.__orig_n_dim_sizes[self.__n_dim_sort_order_orig_s2f].tolist()

        else:
            self.pos_dim_labels = self.__orig_pos_dim_labels.tolist()
            self.spec_dim_labels = self.__orig_spec_dim_labels.tolist()
            self.pos_dim_sizes = self.__orig_pos_dim_sizes.tolist()
            self.spec_dim_sizes = self.__orig_spec_dim_sizes.tolist()
            self.n_dim_labels = self.__orig_n_dim_labs.tolist()
            self.n_dim_sizes = self.__orig_n_dim_sizes.tolist()

    def __set_n_dim_view(self):
        """
        Sets the current view of the N-dimensional form of the dataset
        """

        self.__curr_ndim_form = self.__n_dim_data_s2f if self.__sort_dims else self.__n_dim_data_orig

    @staticmethod
    def __get_anc_labels(h5_dset):
        """
        Takes any dataset which has the labels and units attributes and returns a list of strings
        formatted as 'label k (unit k)'

        Parameters
        ----------
        h5_dset : h5py.Dataset object
            dataset which has labels and units attributes

        Returns
        -------
        labels : list
            list of strings formatted as 'label k (unit k)'
        """
        labels = []
        for lab, unit in zip(get_attr(h5_dset, 'labels'), get_attr(h5_dset, 'units')):
            labels.append('{} ({})'.format(lab, unit))
        return labels

    def get_pos_values(self, dim_name):
        """
        Extract the reference values for the specified position dimension

        Parameters
        ----------
        dim_name : str
            Name of one of the dimensions in `self.pos_dim_labels`

        Returns
        -------
        dim_values : :class:`numpy.ndarray`
            Array containing the unit values of the dimension `dim_name`

        """
        dim_name = validate_single_string_arg(dim_name, 'dim_name')
        return get_unit_values(self.h5_pos_inds, self.h5_pos_vals)[dim_name]

    def get_spec_values(self, dim_name):
        """
        Extract the values for the specified spectroscopic dimension

        Parameters
        ----------
        dim_name : str
            Name of one of the dimensions in `self.spec_dim_labels`

        Returns
        -------
        dim_values : :class:`numpy.ndarray`
            Array containing the unit values of the dimension `dim_name`

        """
        dim_name = validate_single_string_arg(dim_name, 'dim_name')
        return get_unit_values(self.h5_spec_inds, self.h5_spec_vals)[dim_name]

    def get_current_sorting(self):
        """
        Prints the current sorting method.

        """
        if self.__sort_dims:
            print('Data dimensions are sorted in order from fastest changing dimension to slowest.')
        else:
            print('Data dimensions are in the order they occur in the file.')

    def toggle_sorting(self):
        """
        Toggles between sorting from the fastest changing dimension to the slowest and sorting based on the
        order of the labels

        """
        self.__sort_dims = not self.__sort_dims

        self.__set_labels_and_sizes()

        self.__set_n_dim_view()

    def get_n_dim_form(self, as_scalar=False, lazy=False):
        """
        Reshapes the dataset to an N-dimensional array

        Parameters
        ----------
        as_scalar : bool, optional. Default = False
            If False, the data is returned in its original (complex, compound) dtype
            Else, the data is flattened to a real-valued dataset
        lazy : bool, optional. Default = False
            If set to false, n_dim_data will be a :class:`numpy.ndarray`
            Else returned object is :class:`dask.array.core.Array`

        Returns
        -------
        n_dim_data : :class:`numpy.ndarray` or :class:`dask.core.Array`
            N-dimensional form of the dataset

        """

        if self.__curr_ndim_form is None:
            # To be on the safe side, always read as dask Array
            n_dim_data, success = reshape_to_n_dims(self, sort_dims=self.__sort_dims,
                                                    lazy=True)

            if success is not True:
                raise ValueError('Unable to reshape data to N-dimensional form.')
        else:
            n_dim_data = self.__curr_ndim_form

        if as_scalar:
            n_dim_data = flatten_to_real(n_dim_data)

        if not lazy:
            assert(isinstance(n_dim_data, da.core.Array))
            n_dim_data = n_dim_data.compute()

        return n_dim_data

    def __validate_slice_dict(self, slice_dict):
        """
        Validates the slice dictionary

        Parameters
        ----------
        slice_dict : dict
            Dictionary of array-likes.

        Returns
        -------
        None
        """
        if not isinstance(slice_dict, dict):
            raise TypeError('slice_dict should be a dictionary of slice objects')
        for key, val in slice_dict.items():
            # Make sure the dimension is valid
            if key not in self.n_dim_labels:
                raise KeyError('Cannot slice on dimension {}.  '
                               'Valid dimensions are {}.'.format(key, self.n_dim_labels))
            if not isinstance(val, (slice, list, np.ndarray, tuple, int)):
                raise TypeError('The slices must be array-likes or slice objects.')
        return True

    def __slice_n_dim_form(self, slice_dict, verbose=False, lazy=False):
        """
        Slices the N-dimensional form of the dataset based on the slice dictionary.
        Assumes that an N-dimensional form exists and is what was requested

        Parameters
        ----------
        slice_dict : dict
            Dictionary of array-likes. for any dimension one needs to slice
        verbose : bool, optional
            Whether or not to print debugging statements
        lazy : bool, optional. Default = False
            If set to false, data_slice will be a :class:`numpy.ndarray`
            Else returned object is :class:`dask.array.core.Array`

        Returns
        -------
        data_slice : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
            Slice of the dataset.
        success : bool
            Always True
        """
        nd_slice = []

        for dim_name in self.n_dim_labels:
            nd_slice.append(slice_dict.get(dim_name, slice(None)))

        # Dask multidimensional slicing does not work if list is passed:
        nd_slice = tuple(nd_slice)
        if verbose:
            print(self.n_dim_labels)
            print(nd_slice)

        sliced_dset = self.__curr_ndim_form[nd_slice]
        if not lazy:
            sliced_dset = sliced_dset.compute()

        return sliced_dset, True

    def slice(self, slice_dict, ndim_form=True, as_scalar=False, verbose=False, lazy=False):
        """
        Slice the dataset based on an input dictionary of 'str': slice pairs.
        Each string should correspond to a dimension label.  The slices can be
        array-likes or slice objects.

        Parameters
        ----------
        slice_dict : dict
            Dictionary of array-likes. for any dimension one needs to slice
        ndim_form : bool, optional
            Whether or not to return the slice in it's N-dimensional form. Default = True
        as_scalar : bool, optional
            Should the data be returned as scalar values only.
        verbose : bool, optional
            Whether or not to print debugging statements
        lazy : bool, optional. Default = False
            If set to false, data_slice will be a :class:`numpy.ndarray`
            Else returned object is :class:`dask.array.core.Array`

        Returns
        -------
        data_slice : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
            Slice of the dataset.  Dataset has been reshaped to N-dimensions if `success` is True, only
            by Position dimensions if `success` is 'Positions', or not reshape at all if `success`
            is False.
        success : str or bool
            Informs the user as to how the data_slice has been shaped.

        """
        # TODO: Accept sequences of integers and build a list of slice objects for each dimension
        if slice_dict is None:
            slice_dict = dict()
        else:
            self.__validate_slice_dict(slice_dict)

        if not isinstance(as_scalar, bool):
            raise TypeError('as_scalar should be a bool')
        if not isinstance(verbose, bool):
            raise TypeError('verbose should be a bool')

        if self.__n_dim_form_avail and ndim_form:
            return self.__slice_n_dim_form(slice_dict, verbose=verbose, lazy=lazy)

        # Convert the slice dictionary into lists of indices for each dimension
        pos_slice, spec_slice = self._get_pos_spec_slices(slice_dict)
        if verbose:
            print('Position slice: shape - {}'.format(pos_slice.shape))
            print(pos_slice)
            print('Spectroscopic slice: shape - {}'.format(spec_slice.shape))
            print(spec_slice)

        # Now that the slices are built, we just need to apply them to the data
        # This method is slow and memory intensive but shouldn't fail if multiple lists are given.
        if lazy:
            raw_2d = self.__lazy_2d
        else:
            raw_2d = self

        if verbose:
            print('Slicing to 2D based on dataset of shape: {} and type: {}'
                  ''.format(raw_2d.shape, type(raw_2d)))

        if lazy:
            data_slice = raw_2d[pos_slice[:, 0], :][:, spec_slice[:, 0]]
        else:
            if len(pos_slice) <= len(spec_slice):
                # Fewer final positions than spectra
                data_slice = np.atleast_2d(raw_2d[pos_slice[:, 0], :])[:, spec_slice[:, 0]]
            else:
                # Fewer final spectral points compared to positions
                data_slice = np.atleast_2d(raw_2d[:, spec_slice[:, 0]])[pos_slice[:, 0], :]

        if verbose:
            print('data_slice of shape: {} and type: {} after slicing'
                  ''.format(data_slice.shape, type(data_slice)))
        if not lazy:
            orig_shape = data_slice.shape
            data_slice = np.atleast_2d(np.squeeze(data_slice))
            if data_slice.shape[0] == orig_shape[1] and data_slice.shape[1] == orig_shape[0]:
                data_slice = data_slice.T
        if verbose:
            print('data_slice of shape: {} after squeezing'.format(data_slice.shape))

        pos_inds = self.h5_pos_inds[pos_slice, :]
        spec_inds = self.h5_spec_inds[:, spec_slice].reshape([self.h5_spec_inds.shape[0], -1])
        if verbose:
            print('Sliced position indices:')
            print(pos_inds)
            print('Spectroscopic Indices (transposed)')
            print(spec_inds.T)

        # At this point, the empty dimensions MUST be removed in order to avoid problems with dimension sort etc.
        def remove_singular_dims(anc_inds):
            new_inds = []
            for dim_values in anc_inds:
                if len(np.unique(dim_values)) > 1:
                    new_inds.append(dim_values)
            # if all dimensions are removed?
            if len(new_inds) == 0:
                new_inds = np.arange(1)
            else:
                new_inds = np.array(new_inds)
            return new_inds

        pos_inds = np.atleast_2d(remove_singular_dims(pos_inds.T).T)
        spec_inds = np.atleast_2d(remove_singular_dims(spec_inds))

        if verbose:
            print('After removing any singular dimensions')
            print('Sliced position indices:')
            print(pos_inds)
            print('Spectroscopic Indices (transposed)')
            print(spec_inds.T)
            print('data slice of shape: {}. Position indices of shape: {}, Spectroscopic indices of shape: {}'
                  '.'.format(data_slice.shape, pos_inds.shape, spec_inds.shape))

        success = True

        if ndim_form:
            # TODO: if data is already loaded into memory, try to avoid I/O and slice in memory!!!!
            data_slice, success = reshape_to_n_dims(data_slice, h5_pos=pos_inds, h5_spec=spec_inds, verbose=verbose, lazy=lazy)
            data_slice = data_slice.squeeze()

        if as_scalar:
            return flatten_to_real(data_slice), success
        else:
            return data_slice, success

    def _get_pos_spec_slices(self, slice_dict):
        """
        Convert the slice dictionary into two lists of indices, one each for the position and spectroscopic
        dimensions.

        Parameters
        ----------
        slice_dict : dict
            Dictionary of array-likes.

        Returns
        -------
        pos_slice : list of unsigned int
            Position indices included in the slice
        spec_slice : list of unsigned int
            Spectroscopic indices included in the slice
        """
        if slice_dict is None:
            slice_dict = dict()

        self.__validate_slice_dict(slice_dict)

        if len(slice_dict) == 0:
            pos_slice = np.expand_dims(np.arange(self.shape[0]), axis=1)
            spec_slice = np.expand_dims(np.arange(self.shape[1]), axis=1)
            return pos_slice, spec_slice

        # Create default slices that include the entire dimension
        n_dim_slices = dict()
        n_dim_slices_sizes = dict()
        for dim_lab, dim_size in zip(self.n_dim_labels, self.n_dim_sizes):
            n_dim_slices[dim_lab] = list(range(dim_size))
            n_dim_slices_sizes[dim_lab] = len(n_dim_slices[dim_lab])
        # Loop over all the keyword arguments and create slices for each.
        for key, val in slice_dict.items():
            # Check the value and convert to a slice object if possible.
            # Use a list if not.
            if isinstance(val, slice):
                val = n_dim_slices[key][val]
            elif isinstance(val, list):
                pass
            elif isinstance(val, np.ndarray):
                val = val.flatten().tolist()
            elif isinstance(val, tuple):
                val = list(val)
            elif isinstance(val, int):
                val = [val]
            else:
                raise TypeError('The slices must be array-likes or slice objects.')

            if not contains_integers(val, min_val=0):
                # TODO: Is there a more elegant way of handling this?
                raise ValueError('Slicing indices should be >= 0')

            # check to make sure that the values are not out of bounds:
            dim_ind = np.squeeze(np.argwhere(self.__orig_n_dim_labs == key))
            cur_dim_size = self.__orig_n_dim_sizes[dim_ind]
            if np.max(val) >= cur_dim_size:
                raise IndexError('slicing argument for dimension: {} was beyond {}'.format(key, cur_dim_size))

            n_dim_slices[key] = val

            n_dim_slices_sizes[key] = len(val)

        # Build the list of position slice indices
        for pos_ind, pos_lab in enumerate(self.__orig_pos_dim_labels):
            # n_dim_slices[pos_lab] = np.isin(self.h5_pos_inds[:, pos_ind], n_dim_slices[pos_lab])
            temp = [self.h5_pos_inds[:, pos_ind] == item for item in n_dim_slices[pos_lab]]
            n_dim_slices[pos_lab] = np.any(np.vstack(temp), axis=0)
            if pos_ind == 0:
                pos_slice = n_dim_slices[pos_lab]
            else:
                pos_slice = np.logical_and(pos_slice, n_dim_slices[pos_lab])
        pos_slice = np.argwhere(pos_slice)

        # Do the same for the spectroscopic slice
        for spec_ind, spec_lab in enumerate(self.__orig_spec_dim_labels):
            # n_dim_slices[spec_lab] = np.isin(self.h5_spec_inds[spec_ind], n_dim_slices[spec_lab])
            temp = [self.h5_spec_inds[spec_ind] == item for item in n_dim_slices[spec_lab]]
            n_dim_slices[spec_lab] = np.any(np.vstack(temp), axis=0)
            if spec_ind == 0:
                spec_slice = n_dim_slices[spec_lab]
            else:
                spec_slice = np.logical_and(spec_slice, n_dim_slices[spec_lab])
        spec_slice = np.argwhere(spec_slice)

        # TODO: Shouldn't we simply squeeze before returning?
        return pos_slice, spec_slice

    def _get_dims_for_slice(self, slice_dict=None, verbose=False):
        """
        Provides Dimension objects that express the reference position and spectroscopic dimensions for this dataset
        once it is sliced via the provided slicing dictionary.

        Parameters
        ----------
        slice_dict : dict (optional)
            Dictionary to slice one or more dimensions of the dataset by indices
        verbose : bool (optional)
            Whether or not to print debugging statements to stdout. Default = False

        Returns
        -------
        pos_dims : list
            List of :class:`~pyUSID.io.write_utils.Dimension` objects for each of the remaining position dimensions
        spec_dims : list
            List of :class:`~pyUSID.io.write_utils.Dimension` objects for each of the remaining spectroscopic dimensions
        """
        if slice_dict is None:
            slice_dict = dict()

        pos_labels = self.pos_dim_labels
        pos_units = get_attr(self.h5_pos_inds, 'units')
        spec_labels = self.spec_dim_labels
        spec_units = get_attr(self.h5_spec_inds, 'units')

        self.__validate_slice_dict(slice_dict)

        # First work on slicing the ancillary matrices. Determine dimensionality before slicing n dims:
        pos_slices, spec_slices = self._get_pos_spec_slices(slice_dict)
        # Things are too big to print here.

        pos_inds = self.h5_pos_inds[np.squeeze(pos_slices), :]
        pos_vals = self.h5_pos_vals[np.squeeze(pos_slices), :]

        if verbose:
            print('Checking for and correcting the dimensionality of the indices and values datasets:')
            print('Pos Inds: {}, Pos Vals: {}'.format(pos_inds.shape, pos_vals.shape))
        if pos_inds.ndim == 1:
            pos_inds = np.expand_dims(pos_inds, axis=0)
            pos_vals = np.expand_dims(pos_vals, axis=0)

        spec_inds = self.h5_spec_inds[:, np.squeeze(spec_slices)]
        spec_vals = self.h5_spec_vals[:, np.squeeze(spec_slices)]

        if verbose:
            print('Checking for and correcting the dimensionality of the indices and values datasets:')
            print('Spec Inds: {}, Spec Vals: {}'.format(spec_inds.shape, spec_vals.shape))

        if spec_inds.ndim == 1:
            spec_inds = np.expand_dims(spec_inds, axis=1)
            spec_vals = np.expand_dims(spec_vals, axis=1)

        if verbose:
            print('After correction of shape:')
            print('Pos Inds: {}, Pos Vals: {}, Spec Inds: {}, Spec Vals: {}'.format(pos_inds.shape, pos_vals.shape,
                                                                                    spec_inds.shape,
                                                                                    spec_vals.shape))

        # TODO: This assumes an N-dimensional form!
        pos_unit_values = get_unit_values(pos_inds, pos_vals, all_dim_names=self.pos_dim_labels, is_spec=False,
                                          verbose=verbose)
        spec_unit_values = get_unit_values(spec_inds, spec_vals, all_dim_names=self.spec_dim_labels, is_spec=True,
                                           verbose=verbose)

        if verbose:
            print('Position unit values:')
            print(pos_unit_values)
            print('Spectroscopic unit values:')
            print(spec_unit_values)

        # Now unit values will be correct for this slicing

        # additional benefit - remove those dimensions which have at most 1 value:
        def assemble_dimensions(full_labels, full_units, full_values):
            new_labels = []
            new_units = []
            for dim_ind, dim_name in enumerate(full_labels):
                if len(full_values[dim_name]) < 2:
                    del (full_values[dim_name])
                else:
                    new_labels.append(dim_name)
                    new_units.append(full_units[dim_ind])
            return np.array(new_labels), np.array(new_units), full_values

        pos_labels, pos_units, pos_unit_values = assemble_dimensions(pos_labels, pos_units, pos_unit_values)
        spec_labels, spec_units, spec_unit_values = assemble_dimensions(spec_labels, spec_units, spec_unit_values)

        # Ensuring that there are always at least 1 position and spectroscopic dimensions:
        if len(pos_labels) == 0:
            pos_labels = ['arb.']
            pos_units = ['a. u.']
            pos_unit_values = {pos_labels[-1]: np.array([1])}

        if len(spec_labels) == 0:
            spec_labels = ['arb.']
            spec_units = ['a. u.']
            spec_unit_values = {spec_labels[-1]: np.array([1])}

        if verbose:
            print('\n\nAfter removing singular dimensions:')
            print('Position: Labels: {}, Units: {}, Values:'.format(pos_labels, pos_units))
            print(pos_unit_values)
            print('Spectroscopic: Labels: {}, Units: {}, Values:'.format(spec_labels, spec_units))
            print(spec_unit_values)

        pos_dims = []
        for name, units in zip(pos_labels, pos_units):
            pos_dims.append(Dimension(name, units, pos_unit_values[name]))
        spec_dims = []
        for name, units in zip(spec_labels, spec_units):
            spec_dims.append(Dimension(name, units, spec_unit_values[name]))

        return pos_dims, spec_dims

    def slice_to_dataset(self, slice_dict, dset_name=None, verbose=False, **kwargs):
        """
        Slices the dataset, writes its output to back to the HDF5 file, and returns a USIDataset object

        Parameters
        ----------
        slice_dict : dict
            Dictionary to slice one or more dimensions of the dataset by indices
        dset_name : str (optional)
            Name of the new USID Main datset in the HDF5 file that will contain the sliced data.
            Default - the sliced dataset takes the same name as this source dataset
        verbose : bool (optional)
            Whether or not to print debugging statements to stdout. Default = False
        kwargs : keyword arguments
            keyword arguments that will be passed on to write_main_data()

        Returns
        -------
        h5_trunc : USIDataset
            USIDataset containing the sliced data
        """
        if slice_dict is None:
            raise ValueError('slice_dict should not be None or be empty')

        if dset_name is None:
            dset_name = self.name.split('/')[-1]
        else:
            dset_name = validate_single_string_arg(dset_name, 'dset_name')

        if verbose:
            print('Decided / provided name of new sliced HDF5 dataset to be: {}'.format(dset_name))

        pos_dims, spec_dims = self._get_dims_for_slice(slice_dict=slice_dict, verbose=verbose)

        if verbose:
            print('Sliced ancillary datasets returned:\n------------------------------------------')
            print('Position:')
            for dim in pos_dims:
                print(dim)
            print('\nSpectroscopic:')
            for dim in spec_dims:
                print(dim)

        data_slice_2d, success = self.slice(slice_dict, ndim_form=False, as_scalar=False, verbose=verbose)

        if not success:
            raise ValueError('Unable to slice the dataset. success returned: {}'.format(success))

        if verbose:
            print('Slicing the main dataset returned:\n------------------------------------------')
            print('Reshape success: {}'.format(success))
            print('2D data shape: {}'.format(data_slice_2d.shape))

        # check if a pos dimension was sliced:
        pos_sliced = False
        for dim_name in slice_dict.keys():
            if dim_name in self.pos_dim_labels:
                pos_sliced = True
                if verbose:
                    print('Position dimension: {} was sliced'.format(dim_name))
                break
        if not pos_sliced:
            pos_dims = None
            kwargs['h5_pos_inds'] = self.h5_pos_inds
            kwargs['h5_pos_vals'] = self.h5_pos_vals
            if verbose:
                print('Reusing this main datasets position datasets')
        else:
            if verbose:
                print('Using new Position dimensions:\n------------------------------------------')

        spec_sliced = False
        for dim_name in slice_dict.keys():
            if dim_name in self.spec_dim_labels:
                spec_sliced = True
                if verbose:
                    print('Spectroscopic dimension: {} was sliced'.format(dim_name))
                break
        if not spec_sliced:
            spec_dims = None
            kwargs['h5_spec_inds'] = self.h5_spec_inds
            kwargs['h5_spec_vals'] = self.h5_spec_vals
            if verbose:
                print('Reusing this main datasets spectroscopic datasets')
        else:
            if verbose:
                print('Using new spectroscopic dimensions:\n------------------------------------------')

        h5_group = create_results_group(self, 'slice')

        # TODO: Make this memory safe.
        h5_trunc = write_main_dataset(h5_group, data_slice_2d, dset_name, get_attr(self, 'quantity'),
                                      get_attr(self, 'units'), pos_dims, spec_dims, verbose=verbose, **kwargs)
        return h5_trunc

    def visualize(self, slice_dict=None, verbose=False, **kwargs):
        """
        Interactive visualization of this dataset. **Only available on jupyter notebooks**

        Parameters
        ----------
        slice_dict : dictionary, optional
            Slicing instructions
        verbose : bool, optional
            Whether or not to print debugging statements. Default = Off

        Returns
        -------
        fig : :class:`matplotlib.figure` handle
            Handle for the figure object
        axis : :class:`matplotlib.Axes.axis` object
            Axis within which the data was plotted. Note - the interactive visualizer does not return this object
        """

        if slice_dict is None:
            if len(self.pos_dim_labels) > 2 or len(self.spec_dim_labels) > 2:
                raise NotImplementedError('Unable to support visualization of more than 2 position / spectroscopic '
                                          'dimensions. Try slicing the dataset')
            data_slice = self.get_n_dim_form()
            spec_unit_values = get_unit_values(self.h5_spec_inds, self.h5_spec_vals)
            pos_unit_values = get_unit_values(self.h5_pos_inds, self.h5_pos_vals)

            pos_dims = []
            for name, units in zip(self.pos_dim_labels, get_attr(self.h5_pos_inds, 'units')):
                pos_dims.append(Dimension(name, units, pos_unit_values[name]))
            spec_dims = []
            for name, units in zip(self.spec_dim_labels, get_attr(self.h5_spec_inds, 'units')):
                spec_dims.append(Dimension(name, units, spec_unit_values[name]))

        else:
            pos_dims, spec_dims = self._get_dims_for_slice(slice_dict=slice_dict, verbose=verbose)

            # see if the total number of pos and spec keys are either 1 or 2
            if not (0 < len(pos_dims) < 3) or not (0 < len(spec_dims) < 3):
                raise ValueError('Number of position ({}) / spectroscopic dimensions ({}) more than 2'
                                 '. Try slicing again'.format(len(pos_dims), len(spec_dims)))

            # now should be safe to slice:
            data_slice, success = self.slice(slice_dict, ndim_form=True, lazy=False)
            if not success:
                raise ValueError('Something went wrong when slicing the dataset. slice message: {}'.format(success))
            # don't forget to remove singular dimensions via a squeeze
            data_slice = np.squeeze(data_slice)
            # Unlikely event that all dimensions were removed and we are left with a scalar:
            if data_slice.ndim == 0:
                # Nothing to visualize - just return a value
                return data_slice
            # There is a chance that the data dimensionality may have reduced to 1:
            elif data_slice.ndim == 1:
                if len(pos_dims) == 0:
                    data_slice = np.expand_dims(data_slice, axis=0)
                else:
                    data_slice = np.expand_dims(data_slice, axis=-1)

        if verbose:
            print('Position Dimensions:')
            for item in pos_dims:
                print('{}\n{}'.format(len(item.values), item))
            print('Spectroscopic Dimensions:')
            for item in spec_dims:
                print('{}\n{}'.format(len(item.values), item))
            print('N dimensional data sent to visualizer of shape: {}'.format(data_slice.shape))

        # Handle the simple cases first:
        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        def plot_curve(ref_dims, curve):
            x_suffix = ''
            x_exp = get_exponent(ref_dims[0].values)
            if x_exp < -2 or x_exp > 3:
                ref_dims[0].values /= 10 ** x_exp
                x_suffix = ' x $10^{' + str(x_exp) + '}$'

            if is_complex_dtype(curve.dtype):
                # Plot real and image
                fig, axes = plt.subplots(nrows=2, **fig_args)
                for axis, ufunc, comp_name in zip(axes.flat, [np.abs, np.angle], ['Magnitude', 'Phase']):
                    axis.plot(ref_dims[0].values, ufunc(np.squeeze(curve)), **kwargs)
                    if comp_name is 'Magnitude':
                        axis.set_title(self.name + '\n(' + comp_name + ')', pad=15)
                        axis.set_ylabel(self.data_descriptor)
                    else:
                        axis.set_title(comp_name, pad=15)
                        axis.set_ylabel('Phase (rad)')
                        axis.set_xlabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + x_suffix)

                fig.tight_layout()
                return fig, axes
            elif len(curve.dtype) > 0:
                plot_grid = get_plot_grid_size(len(curve.dtype))
                fig, axes = plt.subplots(nrows=plot_grid[0], ncols=plot_grid[1], **fig_args)
                for axis, comp_name in zip(axes.flat, curve.dtype.fields):
                    axis.plot(ref_dims[0].values, np.squeeze(curve[comp_name]), **kwargs)
                    axis.set_title(comp_name, pad=15)
                    axis.set_xlabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + x_suffix)
                    axis.set_ylabel(comp_name)
                # fig.suptitle(self.name)
                fig.tight_layout()
                return fig, axes
            else:
                y_exp = get_exponent(np.squeeze(curve))
                y_suffix = ''
                if y_exp < -2 or y_exp > 3:
                    curve = np.squeeze(curve) / 10 ** y_exp
                    y_suffix = ' x $10^{' + str(y_exp) + '}$'

                fig, axis = plt.subplots(**fig_args)
                axis.plot(ref_dims[0].values, np.squeeze(curve), **kwargs)
                axis.set_xlabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + x_suffix)
                axis.set_ylabel(self.data_descriptor + y_suffix)
                axis.set_title(self.name)

                return fig, axis

        def plot_image(ref_dims, img):
            exponents = [get_exponent(item.values) for item in ref_dims]
            suffix = []
            for item, scale in zip(ref_dims, exponents):
                curr_suff = ''
                if scale < -1 or scale > 3:
                    item.values /= 10 ** scale
                    curr_suff = ' x $10^{' + str(scale) + '}$'
                suffix.append(curr_suff)

            if is_complex_dtype(img.dtype):
                # Plot real and image
                fig, axes = plt.subplots(nrows=2, **fig_args)
                for axis, ufunc, comp_name in zip(axes.flat, [np.abs, np.angle], ['Magnitude', 'Phase']):
                    cbar_label = self.data_descriptor
                    if comp_name is 'Phase':
                        cbar_label = 'Phase (rad)'
                    plot_map(axis, ufunc(np.squeeze(img)), show_xy_ticks=True, show_cbar=True,
                             cbar_label=cbar_label, x_vec=ref_dims[1].values, y_vec=ref_dims[0].values,
                             **kwargs)
                    axis.set_title(self.name + '\n(' + comp_name + ')', pad=15)
                    axis.set_xlabel(ref_dims[1].name + ' (' + ref_dims[1].units + ')' + suffix[1])
                    axis.set_ylabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + suffix[0])
                fig.tight_layout()
                return fig, axes
            elif len(img.dtype) > 0:
                # Compound
                # I would like to have used plot_map_stack by providing it the flattened (real) image cube
                # However, the order of the components in the cube and that provided by img.dtype.fields is not matching
                plot_grid = get_plot_grid_size(len(img.dtype))
                fig, axes = plt.subplots(nrows=plot_grid[0], ncols=plot_grid[1], **fig_args)
                for axis, comp_name in zip(axes.flat, img.dtype.fields):
                    plot_map(axis, np.squeeze(img[comp_name]), show_xy_ticks=True, show_cbar=True,
                             x_vec=ref_dims[1].values, y_vec=ref_dims[0].values, **kwargs)
                    axis.set_title(comp_name, pad=15)
                    axis.set_xlabel(ref_dims[1].name + ' (' + ref_dims[1].units + ')' + suffix[1])
                    axis.set_ylabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + suffix[0])

                # delete empty axes
                for ax_ind in range(len(img.dtype), np.prod(plot_grid)):
                    fig.delaxes(axes.flatten()[ax_ind])

                # fig.suptitle(self.name)
                fig.tight_layout()
                return fig, axes
            else:
                fig, axis = plt.subplots(**fig_args)
                # Need to convert to float since image could be unsigned integers or low precision floats
                plot_map(axis, np.float32(np.squeeze(img)), show_xy_ticks=True, show_cbar=True,
                         cbar_label=self.data_descriptor, x_vec=ref_dims[1].values, y_vec=ref_dims[0].values, **kwargs)
                try:
                    axis.set_title(self.name, pad=15)
                except AttributeError:
                    axis.set_title(self.name)

                axis.set_xlabel(ref_dims[1].name + ' (' + ref_dims[1].units + ')' + suffix[1])
                axis.set_ylabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + suffix[0])
                fig.tight_layout()
                return fig, axis

        if np.prod([len(item.values) for item in spec_dims]) == 1:
            # No spectroscopic dimensions at all
            if len(pos_dims) == 2:
                # 2D spatial map
                # Check if we need to adjust the aspect ratio of the image (only if units are same):
                if pos_dims[0].units == pos_dims[1].units:
                    kwargs['infer_aspect'] = True
                return plot_image(pos_dims, data_slice)
            elif np.prod([len(item.values) for item in pos_dims]) > 1:
                # 1D position curve:
                return plot_curve(pos_dims, data_slice)

        elif np.prod([len(item.values) for item in pos_dims]) == 1:
            if len(spec_dims) == 2:
                # 2D spectrogram
                return plot_image(spec_dims, data_slice)
            elif np.prod([len(item.values) for item in pos_dims]) == 1 and \
                    np.prod([len(item.values) for item in spec_dims]) > 1:
                # 1D spectral curve:
                return plot_curve(spec_dims, data_slice)

        elif len(pos_dims) == 1 and len(spec_dims) == 1 and \
            np.prod([len(item.values) for item in pos_dims]) > 1 and \
            np.prod([len(item.values) for item in spec_dims]) > 1:
            # One spectroscopic and one position dimension
            return plot_image(pos_dims + spec_dims, data_slice)

        # If data has at least one dimension with 2 values in pos. AND spec., it can be visualized interactively:
        return simple_ndim_visualizer(data_slice, pos_dims, spec_dims, verbose=verbose, **kwargs)

    def reduce(self, dims, ufunc=da.mean, to_hdf5=False, dset_name=None, verbose=False):
        """

        Parameters
        ----------
        dims : str or list of str
            Names of the position and/or spectroscopic dimensions that need to be reduced
        ufunc : callable, optional. Default = dask.array.mean
            Reduction function such as dask.array.mean available in dask.array
        to_hdf5 : bool, optional. Default = False
            Whether or not to write the reduced data back to a new dataset
        dset_name : str (optional)
            Name of the new USID Main datset in the HDF5 file that will contain the sliced data.
            Default - the sliced dataset takes the same name as this source dataset
        verbose : bool, optional. Default = False
            Whether or not to print any debugging statements to stdout

        Returns
        -------
        reduced_nd : dask.array object
            Dask array object containing the reduced data.
            Call compute() on this object to get the equivalent numpy array
        h5_main_red : USIDataset
            USIDataset reference if to_hdf5 was set to True. Otherwise - None.
        """
        dims = validate_list_of_strings(dims, 'dims')

        for curr_dim in self.n_dim_labels:
            if curr_dim not in self.n_dim_labels:
                raise KeyError('{} not a dimension in this dataset'.format(curr_dim))

        if ufunc not in [da.all, da.any, da.max, da.mean, da.min, da.moment, da.prod, da.std, da.sum, da.var,
                         da.nanmax, da.nanmean, da.nanmin, da.nanprod, da.nanstd, da.nansum, da.nanvar]:
            raise NotImplementedError('ufunc must be a valid reduction function such as dask.array.mean')

        # At this point, dims are valid
        da_nd, status, labels = reshape_to_n_dims(self, get_labels=True, verbose=verbose, sort_dims=False,
                                                  lazy=True)

        # Translate the names of the dimensions to the indices:
        dim_inds = [np.where(labels == curr_dim)[0][0] for curr_dim in dims]

        # Now apply the reduction:
        reduced_nd = ufunc(da_nd, axis=dim_inds)

        if not to_hdf5:
            return reduced_nd, None

        if dset_name is None:
            dset_name = self.name.split('/')[-1]
        else:
            dset_name = validate_single_string_arg(dset_name, 'dset_name')

        # Create the group to hold the results:

        h5_group = create_results_group(self, 'Reduce')

        # check if a pos dimension was sliced:
        pos_sliced = False
        for dim_name in dims:
            if dim_name in self.pos_dim_labels:
                pos_sliced = True
                if verbose:
                    print('Position dimension: {} was reduced. Breaking...'.format(dim_name))
                break
        if not pos_sliced:
            h5_pos_inds = self.h5_pos_inds
            h5_pos_vals = self.h5_pos_vals
            if verbose:
                print('Reusing this main datasets position datasets')
        else:
            if verbose:
                print('Creating new Position dimensions:\n------------------------------------------')
            # First figure out the names of the position dimensions
            pos_dim_names = []
            for cur_dim in dims:
                if cur_dim in self.pos_dim_labels:
                    pos_dim_names.append(cur_dim)
            if verbose:
                print('Position dimensions reduced: {}'.format(pos_dim_names))

            # Now create the reduced position datasets
            h5_pos_inds, h5_pos_vals = write_reduced_anc_dsets(h5_group, self.h5_pos_inds, self.h5_pos_vals,
                                                               pos_dim_names, is_spec=False, verbose=verbose)

            if verbose:
                print('Position dataset created: {}. Labels: {}'.format(h5_pos_inds, get_attr(h5_pos_inds, 'labels')))

        spec_sliced = False
        for dim_name in dims:
            if dim_name in self.spec_dim_labels:
                spec_sliced = True
                if verbose:
                    print('Spectroscopic dimension: {} was reduced. Breaking...'.format(dim_name))
                break
        if not spec_sliced:
            h5_spec_inds = self.h5_spec_inds
            h5_spec_vals = self.h5_spec_vals
            if verbose:
                print('Reusing this main datasets spectroscopic datasets')
        else:
            if verbose:
                print('Creating new spectroscopic dimensions:\n------------------------------------------')

            # First figure out the names of the position dimensions
            spec_dim_names = []
            for cur_dim in dims:
                if cur_dim in self.spec_dim_labels:
                    spec_dim_names.append(cur_dim)
            if verbose:
                print('Spectroscopic dimensions reduced: {}'.format(spec_dim_names))

            # Now create the reduced position datasets
            h5_spec_inds, h5_spec_vals = write_reduced_anc_dsets(h5_group, self.h5_spec_inds, self.h5_spec_vals,
                                                                 spec_dim_names, is_spec=True, verbose=verbose)

            if verbose:
                print('Spectroscopic dataset created: {}. Labels: {}'.format(h5_spec_inds,
                                                                             get_attr(h5_spec_inds, 'labels')))

                # Now put the reduced N dimensional Dask array back to 2D form:
        reduced_2d, status = reshape_from_n_dims(reduced_nd, h5_pos=h5_pos_inds, h5_spec=h5_spec_inds, verbose=verbose)
        if status != True and verbose:
            print('Status from reshape_from_n_dims: {}'.format(status))
        if verbose:
            print('2D reduced dataset: {}'.format(reduced_2d))

        # Create a HDF5 dataset to hold this flattened 2D data:
        h5_red_main = h5_group.create_dataset(dset_name, shape=reduced_2d.shape,
                                              dtype=reduced_2d.dtype)  # , compression=self.compression)
        if verbose:
            print('Created an empty dataset to hold flattened dataset: {}. Chunks: {}'.format(h5_red_main,
                                                                                              h5_red_main.chunks))

        # Copy the mandatory attributes:
        copy_attributes(self, h5_red_main)

        # Now make this dataset a main dataset:
        link_as_main(h5_red_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)
        if verbose:
            print('{} is a main dataset?: {}'.format(h5_red_main, check_if_main(h5_red_main, verbose=verbose)))

        # Now write this data to the HDF5 dataset:
        if verbose:
            print('About to write dask array to this dataset at path: {}, in file: {}'.format(h5_red_main.name,
                                                                                              self.file.filename))
        reduced_2d.to_hdf5(self.file.filename, h5_red_main.name)

        return reduced_nd, USIDataset(h5_red_main)

    def to_csv(self, output_path=None, force=False):
        """
        Output this USIDataset and position + spectroscopic values to a csv file.
        This should ideally be limited to small datasets only

        Parameters
        ----------
        output_path : str, optional
            path that the output file should be written to.
            By default, the file will be written to the same directory as the HDF5 file
        force : bool, optional
            Whether or not to force large dataset to be written to CSV. Default = False

        Returns
        -------
        output_file: str

        Author - Daniel Streater, Suhas Somnath
        """
        if not isinstance(force, bool):
            raise TypeError('force should be a bool')

        if self.dtype.itemsize * self.size / (1024 ** 2) > 15:
            if force:
                print('Note - the CSV file can be (much) larger than 100 MB')
            else:
                print('CSV file will not be written since the CSV file could be several 100s of MB large.\n'
                      'If you still want the file to be written, add the keyword argument "force=True"\n'
                      'We recommend that you save the data as a .npy or .npz file using numpy.dump')
                return

        if output_path is not None:
            if not isinstance(output_path, str):
                raise TypeError('output_path should be a string with a valid path for the output file')
        else:
            parent_folder, file_name = os.path.split(self.file.filename)
            csv_name = file_name[:file_name.rfind('.')] + self.name.replace('/', '-') + '.csv'
            output_path = os.path.join(parent_folder, csv_name)

        if os.path.exists(output_path):
            if force:
                os.remove(output_path)
            else:
                raise FileExistsError('A file of the following name already exists. Set "force=True" to overwrite.\n'
                                      'File path: ' + output_path)

        header = ''
        for spec_vals_for_dim in self.h5_spec_vals:
            # create one line of the header for each of the spectroscopic dimensions
            header += ','.join(str(item) for item in spec_vals_for_dim) + '\n'
        # Add a dashed-line separating the spec vals from the data
        header += ','.join(
            '--------------------------------------------------------------' for _ in self.h5_spec_vals[0])

        # Write the contents to a temporary file
        np.savetxt('temp.csv', self, delimiter=',', header=header, comments='')

        """
        Create the spectral and position labels for the dataset in string form then
        create the position value array in string form, right-strip the last comma from the 
        string to deliver the correct number of values, append all of the labels and values together,
        save the data and header to a temporary csv output
        """
        # First few lines will have the spectroscopic dimension names + units
        spec_dim_labels = ''
        for dim_desc in self.spec_dim_descriptors:
            spec_dim_labels += ','.join('' for _ in self.pos_dim_labels) + str(dim_desc) + ',\n'

        # Next line will have the position dimension names
        pos_labels = ','.join(pos_dim for pos_dim in self.pos_dim_descriptors) + ',\n'

        # Finally, the remaining rows will have the position values themselves
        pos_values = ''
        for pos_vals_in_row in self.h5_pos_vals:
            pos_values += ','.join(str(item) for item in pos_vals_in_row) + ',\n'
        pos_values = pos_values.rstrip('\n')

        # Now put together all the rows for the first few columns:
        output = spec_dim_labels + pos_labels + pos_values

        left_dset = output.splitlines()

        with open('temp.csv', 'r+') as in_file, open(output_path, 'w') as out_file:
            for left_line, right_line in zip(left_dset, in_file):
                out_file.write(left_line + right_line)

        os.remove('temp.csv')
        print('Successfully wrote this dataset to: ' + output_path)

        return output_path
