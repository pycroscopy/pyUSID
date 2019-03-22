# -*- coding: utf-8 -*-
"""
:class:`~pyUSID.io.numpy_translator.NumpyTranslator` capable of translating numeric arrays to USID HDF5 files

Created on Fri Jan 27 17:58:35 2017

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path, remove
import sys
import h5py
import numpy as np
import dask.array as da
from .write_utils import Dimension
from .translator import Translator, generate_dummy_main_parms
from .hdf_utils import write_main_dataset, write_simple_attrs, create_indexed_group, write_book_keeping_attrs

if sys.version_info.major == 3:
    unicode = str

__all__ = ['ArrayTranslator', 'NumpyTranslator']


class ArrayTranslator(Translator):
    """
    Translator that writes numeric arrays (already in memory) that describe a USID dataset to a HDF5 file
    """

    def translate(self, h5_path, data_name, raw_data, quantity, units, pos_dims, spec_dims,
                  translator_name='NumpyTranslator', parm_dict=None, extra_dsets=None, **kwargs):
        """
        Writes the provided datasets and parameters to an h5 file

        Parameters
        ----------
        h5_path : String / Unicode
            Absolute path of the h5 file to be written
        data_name : String / Unicode
            Name of the scientific data type. Example - 'SEM'
        raw_data : :class:``np.ndarray` or :class:`dask.array.core.Array`
            2D matrix formatted as [position, spectral]
        quantity : String / Unicode
            Name of the physical quantity stored in the dataset. Example - 'Current'
        units : String / Unicode
            Name of units for the quantity stored in the dataset. Example - 'A' for amperes
        pos_dims : Dimension or array-like of Dimension objects
            Sequence of Dimension objects that provides all necessary instructions for constructing the
            indices and values datasets
            Object specifying the instructions necessary for building the Position indices and values datasets
        spec_dims : Dimension or array-like of Dimension objects
            Sequence of Dimension objects that provides all necessary instructions for constructing the
            indices and values datasets
            Object specifying the instructions necessary for building the Spectroscopic indices and values datasets
        translator_name : str / unicode, Optional
            Name of the translator. Example - 'HitachiSEMTranslator'
        parm_dict : dict, Optional
            Dictionary of parameters that will be written under the group 'Measurement_000'
        extra_dsets : dict, Optional
            Dictionary whose values will be written into individual HDF5 datasets and whose corresponding keys provide
            the names of the datasets. You are recommended to limit these to simple and small datasets.
        kwargs: will be passed onto hdf_utils.write_main_dset() which will in turn will be passed onto the creation of
            the dataset. Please pass chunking, compression, dtype, and other arguments this way

        Returns
        -------
        h5_path : String / unicode
            Absolute path of the written h5 file

        """
        for arg, arg_name in zip([h5_path, data_name, translator_name, quantity, units],
                                 ['h5_path', 'data_name', 'translator_name', 'quantity', 'units']):
            if not isinstance(arg, (str, unicode)):
                raise TypeError('{} should be of type - str; was of type - {}'.format(arg_name, type(arg)))
            if len(arg.strip()) == 0:
                raise ValueError('{} should not be an empty string'.format(arg_name))

        if not isinstance(raw_data, (np.ndarray, da.core.Array)):
            raise TypeError('raw_data should either be a np.ndarray or a da.core.Array')

        if raw_data.ndim != 2:
            raise ValueError('raw_data should be a 2-dimensional matrix. Provided array was of shape: {}'
                             '.'.format(raw_data.shape))

        for ind, dimensions in enumerate([pos_dims, spec_dims]):
            if isinstance(dimensions, Dimension):
                dimensions = [dimensions]
            if not isinstance(dimensions, (list, np.ndarray, tuple)):
                raise TypeError('dimensions should be array-like ')
            if not np.all([isinstance(x, Dimension) for x in dimensions]):
                raise TypeError('dimensions should be a sequence of Dimension objects')
            # Check to make sure that the product of the position and spectroscopic dimension sizes match with
            # that of raw_data
            if raw_data.shape[ind] != np.product([len(x.values) for x in dimensions]):
                raise ValueError('Size of dimension[{}] of raw_data not equal to product of size of dimensions: {}'
                                 '.'.format(raw_data.shape[ind], np.product([len(x.values) for x in dimensions])))

        if extra_dsets is not None:
            if not isinstance(extra_dsets, dict):
                raise TypeError('extra_dsets should be specified as dictionaries')
            for key, val in extra_dsets.items():
                if not isinstance(key, (str, unicode)):
                    raise TypeError('keys for extra_dsets should be strings')
                if len(key.strip()) == 0:
                    raise ValueError('keys for extra_dsets should not be empty')
                if np.any([key in x for x in ['Spectroscopic_Indices', 'Spectroscopic_Values', 'Position_Indices',
                                              'Position_Values', 'Raw_Data']]):
                    raise KeyError('keys for extra_dsets cannot match reserved names for existing datasets')
                # Now check for data:
                if not isinstance(val, (list, tuple, np.ndarray, da.core.Array)):
                    raise TypeError('values for extra_dsets should be a tuple, list, or numpy / dask array')
        else:
            extra_dsets = dict()

        if path.exists(h5_path):
            remove(h5_path)

        if parm_dict is None:
            parm_dict = {}

        global_parms = generate_dummy_main_parms()
        global_parms['data_type'] = data_name
        global_parms['translator'] = translator_name

        # Begin writing to file:
        with h5py.File(h5_path) as h5_f:

            # Root attributes first:
            write_simple_attrs(h5_f, global_parms)
            write_book_keeping_attrs(h5_f)

            # measurement group next
            meas_grp = create_indexed_group(h5_f, 'Measurement')
            write_simple_attrs(meas_grp, parm_dict)

            # channel group next
            chan_grp = create_indexed_group(meas_grp, 'Channel')

            _ = write_main_dataset(chan_grp, raw_data, 'Raw_Data', quantity, units, pos_dims, spec_dims, **kwargs)

            for key, val in extra_dsets.items():
                if isinstance(val, da.core.Array):
                    da.to_hdf5(chan_grp.file.filename, {chan_grp + '/' + key: val})
                else:
                    chan_grp.create_dataset(key.strip(), data=val)

        return h5_path


class NumpyTranslator(ArrayTranslator):
    pass
