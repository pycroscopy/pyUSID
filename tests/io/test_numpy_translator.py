# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import os
import sys
import h5py
import numpy as np
from .data_utils import validate_aux_dset_pair
sys.path.append("../../pyUSID/")
from pyUSID.io import NumpyTranslator, write_utils, hdf_utils, USIDataset

if sys.version_info.major == 3:
    unicode = str


class TestNumpyTranslator(unittest.TestCase):

    @staticmethod
    def __delete_existing_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def test_legal_translation(self):
        data_name = 'TestDataType'
        attrs = {'att_1': 'string_val',
                 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4],
                 'att_4': ['str_1', 'str_2', 'str_3']}

        extra_dsets = {'dset_1': np.random.rand(5), 'dset_2': np.arange(25)}

        file_path = 'test_numpy_translator.h5'
        self.__delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']
        pos_dims = []
        for name, unit, length in zip(pos_names, pos_units, pos_sizes):
            pos_dims.append(write_utils.Dimension(name, unit, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for name, unit, length in zip(spec_names, spec_units, spec_sizes):
            spec_dims.append(write_utils.Dimension(name, unit, np.arange(length)))

        spec_data = np.vstack((np.tile(np.arange(7), 2),
                               np.repeat(np.arange(2), 7)))

        translator = NumpyTranslator()
        _ = translator.translate(file_path, data_name, main_data, quantity, units, pos_dims, spec_dims, parm_dict=attrs,
                                 extra_dsets=extra_dsets)

        with h5py.File(file_path, mode='r') as h5_f:
            # we are not interested in most of the attributes under root besides two:
            self.assertEqual(data_name, hdf_utils.get_attr(h5_f, 'data_type'))
            self.assertEqual('NumpyTranslator', hdf_utils.get_attr(h5_f, 'translator'))

            # First level should have absolutely nothing besides one group
            self.assertEqual(len(h5_f.items()), 1)
            self.assertTrue('Measurement_000' in h5_f.keys())
            h5_meas_grp = h5_f['Measurement_000']
            self.assertIsInstance(h5_meas_grp, h5py.Group)

            # check the attributes under this group
            self.assertEqual(len(h5_meas_grp.attrs), len(attrs))
            for key, expected_val in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_meas_grp, key) == expected_val))

            # Again, this group should only have one group - Channel_000
            self.assertEqual(len(h5_meas_grp.items()), 1)
            self.assertTrue('Channel_000' in h5_meas_grp.keys())
            h5_chan_grp = h5_meas_grp['Channel_000']
            self.assertIsInstance(h5_chan_grp, h5py.Group)

            # This channel group is not expected to have any attributes but it will contain the main dataset
            self.assertEqual(len(h5_chan_grp.items()), 5 + len(extra_dsets))
            for dset_name in ['Raw_Data', 'Position_Indices', 'Position_Values', 'Spectroscopic_Indices',
                              'Spectroscopic_Values']:
                self.assertTrue(dset_name in h5_chan_grp.keys())
                h5_dset = h5_chan_grp[dset_name]
                self.assertIsInstance(h5_dset, h5py.Dataset)

            pycro_main = USIDataset(h5_chan_grp['Raw_Data'])

            self.assertIsInstance(pycro_main, USIDataset)
            self.assertEqual(pycro_main.name.split('/')[-1], 'Raw_Data')
            self.assertEqual(pycro_main.parent, h5_chan_grp)
            self.assertTrue(np.allclose(main_data, pycro_main[()]))

            validate_aux_dset_pair(self, h5_chan_grp, pycro_main.h5_pos_inds, pycro_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=pycro_main, is_spectral=False)

            validate_aux_dset_pair(self, h5_chan_grp, pycro_main.h5_spec_inds, pycro_main.h5_spec_vals, spec_names,
                                          spec_units,
                                          spec_data, h5_main=pycro_main, is_spectral=True)

            # Now validate each of the extra datasets:
            for key, val in extra_dsets.items():
                self.assertTrue(key in h5_chan_grp.keys())
                h5_dset = h5_chan_grp[key]
                self.assertIsInstance(h5_dset, h5py.Dataset)
                self.assertTrue(np.allclose(val, h5_dset[()]))

        os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
