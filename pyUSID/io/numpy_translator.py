# -*- coding: utf-8 -*-
"""
:class:`~pyUSID.io.numpy_translator.ArrayTranslator` capable of translating
numeric arrays to USID HDF5 files

Created on Fri Jan 27 17:58:35 2017

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, \
    unicode_literals

from os import path, remove
import sys
import h5py
import numpy as np
import dask.array as da
from .translator import Translator
from .dtype_utils import validate_string_args
from .hdf_utils import write_main_dataset, write_simple_attrs, \
    create_indexed_group, write_book_keeping_attrs, \
    validate_dims_against_main, validate_main_dset
from pyUSID.io.write_utils import validate_dimensions

if sys.version_info.major == 3:
    unicode = str

__all__ = ['ArrayTranslator', 'NumpyTranslator']


class ArrayTranslator(Translator):
    """
    Translator that writes numeric arrays (already in memory) that describe a USID dataset to a HDF5 file
    """

    def translate(self, h5_path, data_name, raw_data, quantity, units, pos_dims, spec_dims,
                  translator_name='ArrayTranslator', parm_dict=None, extra_dsets=None, **kwargs):
        """
        Writes the provided datasets and parameters to an h5 file

        Parameters
        ----------
        h5_path : str
            Absolute path of the h5 file to be written
        data_name : str
            Name of the scientific data type. Example - 'SEM'
        raw_data : :class:`np.ndarray` or :class:`dask.array.core.Array`
            2D matrix formatted as [position, spectral]
        quantity : str
            Name of the physical quantity stored in the dataset. Example - 'Current'
        units : str
            Name of units for the quantity stored in the dataset. Example - 'A' for amperes
        pos_dims : :class:`~pyUSID.io.write_utils.Dimension` or array-like of :class:`~pyUSID.io.write_utils.Dimension`
            objects
            Sequence of :class:`~pyUSID.io.write_utils.Dimension` objects that provides all necessary instructions for
            constructing the indices and values datasets
            Object specifying the instructions necessary for building the Position indices and values datasets
        spec_dims : :class:`~pyUSID.io.write_utils.Dimension` or array-like of :class:`~pyUSID.io.write_utils.Dimension`
            objects
            Sequence of :class:`~pyUSID.io.write_utils.Dimension` objects that provides all necessary instructions for
            constructing the indices and values datasets
            Object specifying the instructions necessary for building the Spectroscopic indices and values datasets
        translator_name : str, Optional
            Name of the translator. Example - 'HitachiSEMTranslator'
        parm_dict : dict, Optional
            Dictionary of parameters that will be written under the group 'Measurement_000'
        extra_dsets : dict, Optional
            Dictionary whose values will be written into individual HDF5 datasets and whose corresponding keys provide
            the names of the datasets. You are recommended to limit these to simple and small datasets.
        kwargs: dict, Optional.
            Additional keyword arguments that will be passed onto :meth:`pyUSID.hdf_utils.write_main_dset()` which will
            in turn will be passed onto the creation of the dataset. Please pass chunking, compression, dtype, and other
            arguments this way

        Returns
        -------
        h5_path : str
            Absolute path of the written h5 file

        """

        h5_path, data_name, translator_name, quantity, units = validate_string_args([h5_path, data_name,
                                                                                     translator_name, quantity, units],
                                                                                    ['h5_path', 'data_name',
                                                                                     'translator_name', 'quantity',
                                                                                     'units'])
        validate_main_dset(raw_data, False)

        for dimensions, dim_name in zip([pos_dims, spec_dims], ['Position', 'Spectroscopic']):
            dimensions = validate_dimensions(dimensions, dim_type=dim_name)
            validate_dims_against_main(raw_data.shape, dimensions, dim_name == 'Spectroscopic')

        if extra_dsets is not None:
            if not isinstance(extra_dsets, dict):
                raise TypeError('extra_dsets should be specified as dictionaries')
            for key, val in extra_dsets.items():
                [key] = validate_string_args(key, 'keys for extra_dsets')
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

        global_parms = dict()
        global_parms['data_type'] = data_name
        global_parms['translator'] = translator_name

        # Begin writing to file:
        with h5py.File(h5_path, mode='w') as h5_f:

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
                    da.to_hdf5(chan_grp.file.filename, {chan_grp.name + '/' + key: val})
                else:
                    chan_grp.create_dataset(key.strip(), data=val)

        return h5_path


class NumpyTranslator(ArrayTranslator):
    pass
