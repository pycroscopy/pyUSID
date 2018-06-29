# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:07:16 2018

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import os
import sys
import h5py
import numpy as np

sys.path.append("../../pyUSID/")
from pyUSID.io import hdf_utils, reg_ref


test_h5_file_path = 'test_reg_ref.h5'

if sys.version_info.major == 3:
    unicode = str


class TestRegRef(unittest.TestCase):

    @staticmethod
    def __delete_existing_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def __write_safe_attrs(h5_object, attrs):
        for key, val in attrs.items():
            h5_object.attrs[key] = val

    @staticmethod
    def __write_string_list_as_attr(h5_object, attrs):
        for key, val in attrs.items():
            h5_object.attrs[key] = np.array(val, dtype='S')

    @staticmethod
    def __write_aux_reg_ref(h5_dset, labels, is_spec=True):
        for index, reg_ref_name in enumerate(labels):
            if is_spec:
                reg_ref_tuple = (slice(index, index + 1), slice(None))
            else:
                reg_ref_tuple = (slice(None), slice(index, index + 1))
            h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]

    @staticmethod
    def __write_main_reg_refs(h5_dset, attrs):
        for reg_ref_name, reg_ref_tuple in attrs.items():
            h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]
        TestRegRef.__write_string_list_as_attr(h5_dset, {'labels': list(attrs.keys())})

    def setUp(self):
        if os.path.exists(test_h5_file_path):
            os.remove(test_h5_file_path)

        with h5py.File(test_h5_file_path) as h5_f:

            h5_raw_grp = h5_f.create_group('Raw_Measurement')
            TestRegRef.__write_safe_attrs(h5_raw_grp, {'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4]})
            TestRegRef.__write_string_list_as_attr(h5_raw_grp, {'att_4': ['str_1', 'str_2', 'str_3']})

            _ = h5_raw_grp.create_group('Misc')

            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_dset_name = 'source_main'
            tool_name = 'Fitter'

            source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

            h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices', data=source_pos_data, dtype=np.uint16)
            TestRegRef.__write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            TestRegRef.__write_string_list_as_attr(h5_pos_inds, pos_attrs)

            # make the values more interesting:
            source_pos_data = np.vstack((source_pos_data[:, 0] * 50, source_pos_data[:, 1] * 1.25)).T

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            TestRegRef.__write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            TestRegRef.__write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''], 'labels': ['Bias', 'Cycle']}

            h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                            dtype=np.uint16)
            TestRegRef.__write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
            TestRegRef.__write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack(
                (np.tile(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)),
                         num_cycles),
                 np.repeat(np.arange(num_cycles), num_cycle_pts)))

            h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                            dtype=np.float32)
            TestRegRef.__write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
            TestRegRef.__write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
            TestRegRef.__write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})
            TestRegRef.__write_main_reg_refs(h5_source_main, {'even_rows': (slice(0, None, 2), slice(None)),
                                                   'odd_rows': (slice(1, None, 2), slice(None))})

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals]:
                h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

            _ = h5_raw_grp.create_dataset('Ancillary', data=np.arange(5))

            # Now add a few results:

            h5_results_grp_1 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_000')
            TestRegRef.__write_safe_attrs(h5_results_grp_1, {'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4]})
            TestRegRef.__write_string_list_as_attr(h5_results_grp_1, {'att_4': ['str_1', 'str_2', 'str_3']})

            num_cycles = 1
            num_cycle_pts = 7

            results_spec_inds = np.expand_dims(np.arange(num_cycle_pts), 0)
            results_spec_attrs = {'units': ['V'], 'labels': ['Bias']}

            h5_results_1_spec_inds = h5_results_grp_1.create_dataset('Spectroscopic_Indices',
                                                                     data=results_spec_inds, dtype=np.uint16)
            TestRegRef.__write_aux_reg_ref(h5_results_1_spec_inds, results_spec_attrs['labels'], is_spec=True)
            TestRegRef.__write_string_list_as_attr(h5_results_1_spec_inds, results_spec_attrs)

            results_spec_vals = np.expand_dims(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)), 0)

            h5_results_1_spec_vals = h5_results_grp_1.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                     dtype=np.float32)
            TestRegRef.__write_aux_reg_ref(h5_results_1_spec_vals, results_spec_attrs['labels'], is_spec=True)
            TestRegRef.__write_string_list_as_attr(h5_results_1_spec_vals, results_spec_attrs)

            results_1_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_results_1_main = h5_results_grp_1.create_dataset('results_main', data=results_1_main_data)
            TestRegRef.__write_safe_attrs(h5_results_1_main, {'units': 'pF', 'quantity': 'Capacitance'})

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_1_spec_inds, h5_results_1_spec_vals]:
                h5_results_1_main.attrs[dset.name.split('/')[-1]] = dset.ref

            # add another result with different parameters

            h5_results_grp_2 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_001')
            TestRegRef.__write_safe_attrs(h5_results_grp_2, {'att_1': 'other_string_val', 'att_2': 5.4321, 'att_3': [4, 1, 3]})
            TestRegRef.__write_string_list_as_attr(h5_results_grp_2, {'att_4': ['s', 'str_2', 'str_3']})

            results_2_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_results_2_main = h5_results_grp_2.create_dataset('results_main', data=results_2_main_data)
            TestRegRef.__write_safe_attrs(h5_results_2_main, {'units': 'pF', 'quantity': 'Capacitance'})

            h5_results_2_spec_inds = h5_results_grp_2.create_dataset('Spectroscopic_Indices',
                                                                     data=results_spec_inds, dtype=np.uint16)
            TestRegRef.__write_aux_reg_ref(h5_results_2_spec_inds, results_spec_attrs['labels'], is_spec=True)
            TestRegRef.__write_string_list_as_attr(h5_results_2_spec_inds, results_spec_attrs)

            h5_results_2_spec_vals = h5_results_grp_2.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                     dtype=np.float32)
            TestRegRef.__write_aux_reg_ref(h5_results_2_spec_vals, results_spec_attrs['labels'], is_spec=True)
            TestRegRef.__write_string_list_as_attr(h5_results_2_spec_vals, results_spec_attrs)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_2_spec_inds, h5_results_2_spec_vals]:
                h5_results_2_main.attrs[dset.name.split('/')[-1]] = dset.ref

    def tearDown(self):
        os.remove(test_h5_file_path)

    def test_get_region_illegal_01(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(KeyError):
                reg_ref.get_region(h5_f['/Raw_Measurement/source_main'], 'non_existent')

    def test_get_region_legal_01(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_source = h5_f['/Raw_Measurement/source_main']
            returned = reg_ref.get_region(h5_source, 'even_rows')
            self.assertTrue(np.all(returned == h5_source[range(0, h5_source.shape[0], 2)]))