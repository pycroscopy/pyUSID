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


sys.path.append("../../pyUSID/")
from pyUSID.io import USIDataset, hdf_utils

from . import data_utils

if sys.version_info.major == 3:
    unicode = str

test_h5_file_path = 'test_usi_dataset.h5'


class TestUSIDataset(unittest.TestCase):

    def setUp(self):

        if os.path.exists(test_h5_file_path):
            os.remove(test_h5_file_path)
        with h5py.File(test_h5_file_path) as h5_f:

            h5_raw_grp = h5_f.create_group('Raw_Measurement')
            data_utils.write_safe_attrs(h5_raw_grp, {'att_1': 'string_val', 'att_2': 1.2345,
                                                             'att_3': [1, 2, 3, 4]})
            data_utils.write_string_list_as_attr(h5_raw_grp, {'att_4': ['str_1', 'str_2', 'str_3']})

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
            data_utils.write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_inds, pos_attrs)

            # make the values more interesting:
            source_pos_data = np.vstack((source_pos_data[:, 0] * 50, source_pos_data[:, 1] * 1.25)).T

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_spec_data = np.vstack((np.repeat(np.arange(num_cycle_pts), num_cycles),
                                          np.tile(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''], 'labels': ['Bias', 'Cycle']}

            h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                            dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack(
                (np.repeat(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)),
                         num_cycles),
                 np.tile(np.arange(num_cycles), num_cycle_pts)))

            h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                            dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
            data_utils.write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})
            data_utils.write_main_reg_refs(h5_source_main, {'even_rows': (slice(0, None, 2), slice(None)),
                                                                    'odd_rows': (slice(1, None, 2), slice(None))})

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals]:
                h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

            _ = h5_raw_grp.create_dataset('Ancillary', data=np.arange(5))

            # Now add a few results:

            h5_results_grp_1 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_000')
            data_utils.write_safe_attrs(h5_results_grp_1,
                                                {'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4]})
            data_utils.write_string_list_as_attr(h5_results_grp_1, {'att_4': ['str_1', 'str_2', 'str_3']})

            num_cycles = 1
            num_cycle_pts = 7

            results_spec_inds = np.expand_dims(np.arange(num_cycle_pts), 0)
            results_spec_attrs = {'units': ['V'], 'labels': ['Bias']}

            h5_results_1_spec_inds = h5_results_grp_1.create_dataset('Spectroscopic_Indices',
                                                                     data=results_spec_inds, dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_results_1_spec_inds, results_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_results_1_spec_inds, results_spec_attrs)

            results_spec_vals = np.expand_dims(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)), 0)

            h5_results_1_spec_vals = h5_results_grp_1.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                     dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_results_1_spec_vals, results_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_results_1_spec_vals, results_spec_attrs)

            results_1_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_results_1_main = h5_results_grp_1.create_dataset('results_main', data=results_1_main_data)
            data_utils.write_safe_attrs(h5_results_1_main, {'units': 'pF', 'quantity': 'Capacitance'})

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_1_spec_inds, h5_results_1_spec_vals]:
                h5_results_1_main.attrs[dset.name.split('/')[-1]] = dset.ref

            # add another result with different parameters

            h5_results_grp_2 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_001')
            data_utils.write_safe_attrs(h5_results_grp_2,
                                                {'att_1': 'other_string_val', 'att_2': 5.4321, 'att_3': [4, 1, 3]})
            data_utils.write_string_list_as_attr(h5_results_grp_2, {'att_4': ['s', 'str_2', 'str_3']})

            results_2_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_results_2_main = h5_results_grp_2.create_dataset('results_main', data=results_2_main_data)
            data_utils.write_safe_attrs(h5_results_2_main, {'units': 'pF', 'quantity': 'Capacitance'})

            h5_results_2_spec_inds = h5_results_grp_2.create_dataset('Spectroscopic_Indices',
                                                                     data=results_spec_inds, dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_results_2_spec_inds, results_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_results_2_spec_inds, results_spec_attrs)

            h5_results_2_spec_vals = h5_results_grp_2.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                     dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_results_2_spec_vals, results_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_results_2_spec_vals, results_spec_attrs)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_2_spec_inds, h5_results_2_spec_vals]:
                h5_results_2_main.attrs[dset.name.split('/')[-1]] = dset.ref

    def tearDown(self):
        os.remove(test_h5_file_path)

    def test_string_representation(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            usi_dset = USIDataset(h5_main)
            actual = usi_dset.__repr__()
            actual = [line.strip() for line in actual.split("\n")]
            actual = [actual[line_ind] for line_ind in [0, 2, 4, 7, 8, 10, 11]]

            expected = list()
            expected.append(h5_main.__repr__())
            expected.append(h5_main.name)
            expected.append(hdf_utils.get_attr(h5_main, "quantity") + " (" + hdf_utils.get_attr(h5_main, "units") + ")")
            for h5_inds in [usi_dset.h5_pos_inds, usi_dset.h5_spec_inds]:
                for dim_name, dim_size in zip(hdf_utils.get_attr(h5_inds, "labels"),
                                              hdf_utils.get_dimensionality(h5_inds)):
                    expected.append(dim_name + ' - size: ' + str(dim_size))
            self.assertTrue(np.all([x == y for x, y in zip(actual, expected)]))


class TestEquality(TestUSIDataset):

    def test_correct_USIDataset(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = USIDataset(h5_main)
            self.assertTrue(expected == expected)

    def test_correct_h5_dataset(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = USIDataset(h5_main)
            self.assertTrue(expected == h5_main)

    def test_incorrect_USIDataset(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = USIDataset(h5_main)
            incorrect = USIDataset(h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'])
            self.assertFalse(expected == incorrect)

    def test_incorrect_h5_dataset(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = USIDataset(h5_main)
            incorrect = h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices']
            self.assertFalse(expected == incorrect)

    def test_incorrect_object(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = USIDataset(h5_main)
            incorrect = np.zeros(shape=(1, 2, 3, 4))
            self.assertFalse(expected == incorrect)


class TestGetNDimFormExists(TestUSIDataset):

    def test_unsorted(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = np.reshape(h5_main, (3, 5, 7, 2))
            expected = np.transpose(expected, (1, 0, 2, 3))
            usi_dset = USIDataset(h5_main)
            self.assertTrue(np.allclose(expected, usi_dset.get_n_dim_form(lazy=False)))

    def test_sorted(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = np.reshape(h5_main, (3, 5, 7, 2))
            expected = np.transpose(expected, (1, 0, 3, 2))
            usi_dset = USIDataset(h5_main)
            usi_dset.toggle_sorting()
            self.assertTrue(np.allclose(expected, usi_dset.get_n_dim_form(lazy=False)))


class TestPosSpecSlices(TestUSIDataset):

    def test_empty_dict(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = usi_main._get_pos_spec_slices({})
            self.assertTrue(np.allclose(np.expand_dims(np.arange(14), axis=1), actual_spec))
            self.assertTrue(np.allclose(np.expand_dims(np.arange(15), axis=1), actual_pos))

    def test_non_existent_dim(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = usi_main._get_pos_spec_slices({'blah': 4, 'X': 3, 'Y': 1})

    def test_incorrect_type(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(TypeError):
                _ = usi_main._get_pos_spec_slices({'X': 'fdfd', 'Y': 1})

    def test_negative_index(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = usi_main._get_pos_spec_slices({'X': -4, 'Y': 1})

    def test_out_of_bounds(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = usi_main._get_pos_spec_slices({'X': 15, 'Y': 1})

    def test_one_pos_dim_removed(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            # orig_pos = np.vstack([np.tile(np.arange(5), 3), np.repeat(np.arange(3), 5)]).T
            # orig_spec = np.vstack([np.tile(np.arange(7), 2), np.repeat(np.arange(2), 7)])
            actual_pos, actual_spec = usi_main._get_pos_spec_slices({'X': 3})
            # we want every fifth position starting from 3
            expected_pos = np.expand_dims(np.arange(3, 15, 5), axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_one_pos_dim_sliced(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = usi_main._get_pos_spec_slices({'X': slice(1, 5, 2)})
            # we want every fifth position starting from 3
            positions = []
            for row_ind in range(3):
                for col_ind in range(1, 5, 2):
                    positions.append(5 * row_ind + col_ind)
            expected_pos = np.expand_dims(positions, axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_two_pos_dim_sliced(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = usi_main._get_pos_spec_slices({'X': slice(1, 5, 2), 'Y': 1})
            # we want every fifth position starting from 3
            positions = []
            for row_ind in range(1, 2):
                for col_ind in range(1, 5, 2):
                    positions.append(5 * row_ind + col_ind)
            expected_pos = np.expand_dims(positions, axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_two_pos_dim_sliced_list(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = usi_main._get_pos_spec_slices({'X': [1, 2, 4], 'Y': 1})
            # we want every fifth position starting from 3
            positions = []
            for row_ind in range(1, 2):
                for col_ind in [1, 2, 4]:
                    positions.append(5 * row_ind + col_ind)
            expected_pos = np.expand_dims(positions, axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_both_pos_removed(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = usi_main._get_pos_spec_slices({'X': 3, 'Y': 1})
            # we want every fifth position starting from 3
            expected_pos = np.expand_dims([1 * 5 + 3], axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_pos_and_spec_sliced_list(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            h5_pos_inds = usi_main.h5_pos_inds
            h5_spec_inds = usi_main.h5_spec_inds
            actual_pos, actual_spec = usi_main._get_pos_spec_slices({'X': [1, 2, 4], 'Bias': slice(1, 7, 3)})
            # we want every fifth position starting from 3
            positions = []
            for col_ind in [1, 2, 4]:
                positions += np.argwhere(h5_pos_inds[h5_pos_inds.attrs['X']] == col_ind)[:, 0].tolist()
            specs = []
            for bias_ind in range(1, 7, 3):
                specs += np.argwhere(h5_spec_inds[h5_spec_inds.attrs['Bias']] == bias_ind)[:, 1].tolist()
            expected_pos = np.expand_dims(positions, axis=1)
            expected_spec = np.expand_dims(specs, axis=1)
            expected_pos.sort(axis=0)
            expected_spec.sort(axis=0)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))


class TestGetUnitValues(TestUSIDataset):

    def test_get_pos_values(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            expected = usi_main.h5_pos_vals[:5, 0]
            actual = usi_main.get_pos_values('X')
            self.assertTrue(np.allclose(expected, actual))
            expected = usi_main.h5_pos_vals[0:None:5, 1]
            actual = usi_main.get_pos_values('Y')
            self.assertTrue(np.allclose(expected, actual))

    def test_get_pos_values_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = usi_main.get_pos_values('blah')
            with self.assertRaises(TypeError):
                _ = usi_main.get_pos_values(np.array(5))

    def test_get_spec_values(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            expected = usi_main.h5_spec_vals[0, ::2]
            actual = usi_main.get_spec_values('Bias')
            self.assertTrue(np.allclose(expected, actual))
            expected = usi_main.h5_spec_vals[1, 0:None:7]
            actual = usi_main.get_spec_values('Cycle')
            self.assertTrue(np.allclose(expected, actual))

    def test_get_spec_values_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = usi_main.get_spec_values('blah')
            with self.assertRaises(TypeError):
                _ = usi_main.get_spec_values(np.array(5))


class TestSlice(TestUSIDataset):

    def test_empty(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = usi_main.slice(None, lazy=False)
            expected = np.transpose(np.reshape(usi_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            self.assertTrue(np.allclose(expected, actual))

    def test_non_existent_dim(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = usi_main.slice({'blah': 4, 'X': 3, 'Y': 1})

    def test_incorrect_type(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(TypeError):
                _ = usi_main.slice({'X': 'fdfd', 'Y': 1})

    def test_negative_index(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = usi_main.slice({'X': -4, 'Y': 1})

    def test_out_of_bounds(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = usi_main.slice({'X': 15, 'Y': 1})

    def test_one_pos_dim_removed(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = usi_main.slice(slice_dict={'X': 3}, lazy=False)
            n_dim_form = np.transpose(np.reshape(usi_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[3, :, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_one_pos_dim_sliced(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = usi_main.slice({'X': slice(1, 5, 2)}, lazy=False)
            n_dim_form = np.transpose(np.reshape(usi_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[slice(1, 5, 2), :, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_two_pos_dim_sliced(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = usi_main.slice({'X': slice(1, 5, 2), 'Y': 1}, lazy=False)
            n_dim_form = np.transpose(np.reshape(usi_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[slice(1, 5, 2), 1, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_two_pos_dim_sliced_list(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = usi_main.slice({'X': [1, 2, 4], 'Y': 1}, lazy=False)
            n_dim_form = np.transpose(np.reshape(usi_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[[1, 2, 4], 1, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_both_pos_removed(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = usi_main.slice({'X': 3, 'Y': 1}, lazy=False)
            n_dim_form = np.transpose(np.reshape(usi_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[3, 1, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_pos_and_spec_sliced_list(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = usi_main.slice({'X': [1, 2, 4], 'Bias': slice(1, 7, 3)}, lazy=False)
            n_dim_form = np.transpose(np.reshape(usi_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[[1, 2, 4], :, slice(1, 7, 3), :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_all_dims_sliced_list(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = usi_main.slice({'X': [1, 2, 4], 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1},
                                               lazy=False)
            n_dim_form = np.transpose(np.reshape(usi_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[[1, 2, 4], 2, slice(1, 7, 3), 1]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)


class TestSorting(TestUSIDataset):

    def test_toggle_sorting(self):
        # Need to change data file so that sorting actually does something
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])

            self.assertTrue(usi_main.n_dim_labels == ['X', 'Y', 'Bias', 'Cycle'])

            usi_main.toggle_sorting()

            self.assertTrue(usi_main.n_dim_labels==['X', 'Y', 'Cycle', 'Bias'])

    def test_get_current_sorting(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            unsorted_str = 'Data dimensions are in the order they occur in the file.\n'
            sorted_str = 'Data dimensions are sorted in order from fastest changing dimension to slowest.\n'
            # Initial state should be unsorted
            self.assertFalse(usi_main._USIDataset__sort_dims)
            with data_utils.capture_stdout() as get_value:
                usi_main.get_current_sorting()
                test_str = get_value()
            self.assertTrue(test_str == unsorted_str)
            # Toggle sorting.  Sorting should now be true.
            usi_main.toggle_sorting()
            self.assertTrue(usi_main._USIDataset__sort_dims)
            with data_utils.capture_stdout() as get_value:
                usi_main.get_current_sorting()
                test_str = get_value()
            self.assertTrue(test_str == sorted_str)


if __name__ == '__main__':
    unittest.main()
