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
import dask.array as da
import shutil

sys.path.append("../../pyUSID/")
from pyUSID.io import hdf_utils, write_utils, USIDataset

from . import data_utils


if sys.version_info.major == 3:
    unicode = str


class TestHDFUtils(unittest.TestCase):

    def setUp(self):
        data_utils.make_beps_file()
        data_utils.make_sparse_sampling_file()
        data_utils.make_incomplete_measurement_file()
        data_utils.make_relaxation_file()

    def tearDown(self):
        for file_path in [data_utils.std_beps_path, 
                          data_utils.sparse_sampling_path,
                          data_utils.incomplete_measurement_path,
                          data_utils.relaxation_path]:
            data_utils.delete_existing_file(file_path)

    def test_get_dimensionality_legal_no_sort(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_shapes = [[7, 2],
                               [7],
                               [5, 3]]
            for h5_dset, exp_shape in zip(h5_dsets, expected_shapes):
                self.assertTrue(np.all(exp_shape == hdf_utils.get_dimensionality(h5_dset)))

    def test_get_dimensionality_legal_w_sort(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_shapes = [[2, 7],
                               [7],
                               [3, 5]]
            sort_orders = [[1, 0],
                           [0],
                           [1, 0]]
            for h5_dset, s_oder, exp_shape in zip(h5_dsets, sort_orders, expected_shapes):
                self.assertTrue(np.all(exp_shape == hdf_utils.get_dimensionality(h5_dset, index_sort=s_oder)))

    def test_check_is_main_legal_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/results_main']]
            for dset in expected_dsets:
                self.assertTrue(hdf_utils.check_if_main(dset))

    def test_check_is_main_illegal_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            not_main_dsets = [h5_f,
                              4.123,
                              np.arange(6),
                              h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/Spectroscopic_Values']]
            for dset in not_main_dsets:
                self.assertFalse(hdf_utils.check_if_main(dset))

    def test_get_sort_order_simple(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_order = [[0, 1], [0], [0, 1]]
            for h5_dset, exp_order in zip(h5_dsets, expected_order):
                self.assertTrue(np.all(exp_order == hdf_utils.get_sort_order(h5_dset)))

    def test_get_sort_order_reversed(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_dsets = [np.flipud(h5_f['/Raw_Measurement/Spectroscopic_Indices']),
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        np.fliplr(h5_f['/Raw_Measurement/Position_Indices'])]
            expected_order = [[1, 0], [0], [1, 0]]
            for h5_dset, exp_order in zip(h5_dsets, expected_order):
                self.assertTrue(np.all(exp_order == hdf_utils.get_sort_order(h5_dset)))

    def test_get_source_dataset_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_groups = [h5_f['/Raw_Measurement/source_main-Fitter_000'],
                        h5_f['/Raw_Measurement/source_main-Fitter_001']]
            h5_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            for h5_grp in h5_groups:
                self.assertEqual(h5_main, hdf_utils.get_source_dataset(h5_grp))

    def test_get_source_dataset_illegal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(ValueError):
                _ = hdf_utils.get_source_dataset(h5_f['/Raw_Measurement/Misc'])

    def test_get_unit_values_source_spec_all(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float32(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False))),
                        'Cycle': [0., 1.]}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals)
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_source_spec_all_explicit(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float32(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False))),
                        'Cycle': [0., 1.]}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Bias'])
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_illegal_key(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Does not exist'])

    def test_get_unit_values_illegal_dset(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Ancillary']
            with self.assertRaises(ValueError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Bias'])

    def test_get_unit_values_source_spec_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float32(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)))}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names='Bias')
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_source_pos_all(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_vals = h5_f['/Raw_Measurement/Position_Values']
            num_rows = 3
            num_cols = 5
            expected = {'X': np.float32(np.arange(num_cols) * 50),
                        'Y': np.float32(np.arange(num_rows) * 1.25)}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals)
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_source_pos_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_vals = h5_f['/Raw_Measurement/Position_Values']
            num_rows = 3
            expected = {'Y': np.float16(np.arange(num_rows) * 1.25)}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names='Y')
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_all_dim_names_not_provided(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices'][()]
            h5_vals = h5_f['/Raw_Measurement/Position_Values'][()]

            with self.assertRaises(TypeError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Y'])

    def test_get_unit_values_dependent_dim(self):
        with h5py.File(data_utils.relaxation_path, mode='r') as h5_f:
            h5_inds = h5_f['/Measurement_000/Channel_000/Spectroscopic_Indices']
            h5_vals = h5_f['/Measurement_000/Channel_000/Spectroscopic_Values']
            spec_dim_names = hdf_utils.get_attr(h5_inds, 'labels')
            ret_dict = hdf_utils.get_unit_values(h5_inds, h5_vals)
            for dim_ind, dim_name in enumerate(spec_dim_names):
                exp_val = hdf_utils.get_attr(h5_inds, 'unit_vals_dim_' + str(dim_ind))
                act_val = ret_dict[dim_name]
                self.assertTrue(np.allclose(exp_val, act_val))

    def test_get_unit_values_sparse_samp_no_attr(self):
        # What should the user expect this function to do? throw an error.
        # Without the attribute, this function will have no idea that it is looking at a sparse sampling case
        # it will return the first and second columns of vals blindly
        with h5py.File(data_utils.sparse_sampling_path, mode='r') as h5_f:
            h5_inds = h5_f['/Measurement_000/Channel_000/Position_Indices']
            h5_vals = h5_f['/Measurement_000/Channel_000/Position_Values']
            dim_names = hdf_utils.get_attr(h5_inds, 'labels')
            ret_dict = hdf_utils.get_unit_values(h5_inds, h5_vals)
            for dim_ind, dim_name in enumerate(dim_names):
                exp_val = h5_vals[:, dim_ind]
                act_val = ret_dict[dim_name]
                self.assertTrue(np.allclose(exp_val, act_val))

    def test_get_unit_values_sparse_samp_w_attr(self):
        # What should the user expect this function to do? throw an error.
        with h5py.File(data_utils.sparse_sampling_path, mode='r') as h5_f:
            h5_inds = h5_f['/Measurement_000/Channel_001/Position_Indices']
            h5_vals = h5_f['/Measurement_000/Channel_001/Position_Values']

            with self.assertRaises(ValueError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Y'])

    def test_get_unit_values_incomp_dim_no_attr(self):
        # What should the user expect this function to do? throw an error.
        # Given that the unit values for each tile are different, it should throw a ValueError for X.
        # Even though we know Y is incomplete, it won't know since it wasn't looking at X.
        # However, now this function will automatically find unit values for ALL dimensions just to catch such scenarios
        with h5py.File(data_utils.incomplete_measurement_path, mode='r') as h5_f:
            h5_inds = h5_f['/Measurement_000/Channel_000/Position_Indices']
            h5_vals = h5_f['/Measurement_000/Channel_000/Position_Values']

            with self.assertRaises(ValueError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals)

            with self.assertRaises(ValueError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['X'])

            with self.assertRaises(ValueError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Y'])

    def test_reshape_to_n_dims_h5_no_sort_reqd(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main, get_labels=True, sort_dims=False,
                                                                 as_dask_array=False)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['X', 'Y', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(h5_main[()], (num_rows, num_cols, num_cycles, num_cycle_pts))
            expected_n_dim = np.transpose(expected_n_dim, (1, 0, 3, 2))
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main, get_labels=True, sort_dims=True,
                                                                 as_dask_array=False)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['X', 'Y', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(h5_main[()], (num_rows, num_cols, num_cycles, num_cycle_pts))
            expected_n_dim = np.transpose(expected_n_dim, (1, 0, 3, 2))
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

    def test_reshape_to_n_dims_h5_not_main_dset(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/Ancillary']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            h5_spec = h5_f['/Raw_Measurement/Spectroscopic_Indices']

            # Not main
            with self.assertRaises(ValueError):
                _ = hdf_utils.reshape_to_n_dims(h5_main)

            # Not main and not helping that we are supplign incompatible ancillary datasets
            with self.assertRaises(ValueError):
                _ = hdf_utils.reshape_to_n_dims(h5_main, h5_pos=h5_pos, h5_spec=h5_spec)

            # main but we are supplign incompatible ancillary datasets
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000/results_main']
            with self.assertRaises(ValueError):
                _ = hdf_utils.reshape_to_n_dims(h5_main, h5_pos=h5_pos, h5_spec=h5_spec)

    def test_reshape_to_n_dim_numpy(self):
        num_rows = 3
        num_cols = 5
        num_cycles = 2
        num_cycle_pts = 7
        # arrange as slow, fast instead of fast, slow
        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T

        source_main_data = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
        for row_ind in range(num_rows):
            for col_ind in range(num_cols):
                for cycle_ind in range(num_cycles):
                    for bias_ind in range(num_cycle_pts):
                        val = 1E+3*row_ind + 1E+2*col_ind + 1E+1*cycle_ind + bias_ind
                        source_main_data[row_ind*num_cols + col_ind, cycle_ind*num_cycle_pts + bias_ind] = val

        # make spectroscopic slow, fast instead of fast, slow
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles)))
        n_dim, success = hdf_utils.reshape_to_n_dims(source_main_data, h5_pos=source_pos_data,
                                                     h5_spec=source_spec_data, get_labels=False, as_dask_array=False)
        expected_n_dim = np.reshape(source_main_data, (num_rows, num_cols, num_cycles, num_cycle_pts))
        self.assertTrue(np.allclose(expected_n_dim, n_dim))

    def test_reshape_to_n_dim_sort_required(self):
        file_path = 'reshape_to_n_dim_sort_required.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_raw_grp = h5_f.create_group('Raw_Measurement')

            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_dset_name = 'source_main'

            # arrange as slow, fast instead of fast, slow
            source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                         np.tile(np.arange(num_cols), num_rows))).T
            pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

            h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices', data=source_pos_data, dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_inds, pos_attrs)

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_main_data = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
            for row_ind in range(num_rows):
                for col_ind in range(num_cols):
                    for cycle_ind in range(num_cycles):
                        for bias_ind in range(num_cycle_pts):
                            val = 1E+3*row_ind + 1E+2*col_ind + 1E+1*cycle_ind + bias_ind
                            source_main_data[row_ind*num_cols + col_ind, cycle_ind*num_cycle_pts + bias_ind] = val

            # source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
            data_utils.write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})

            # make spectroscopic slow, fast instead of fast, slow
            source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                          np.tile(np.arange(num_cycle_pts), num_cycles)))
            source_spec_attrs = {'units': ['', 'V'], 'labels': ['Cycle', 'Bias']}

            h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                            dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

            h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                            dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals]:
                h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_source_main, get_labels=True, sort_dims=True,
                                                                 as_dask_array=False)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['Y', 'X', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(source_main_data, (num_rows, num_cols, num_cycles, num_cycle_pts))
            expected_n_dim = np.transpose(expected_n_dim, [1, 0, 3, 2])
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

        os.remove(file_path)

    def test_reshape_from_n_dims_pos_and_spec_provided(self):
        num_rows = 3
        num_cols = 5
        num_cycles = 2
        num_cycle_pts = 7

        # the N dimensional dataset should be arranged in the following order:
        # [positions slowest to fastest, spectroscopic slowest to fastest]
        source_nd = np.zeros(shape=(num_rows, num_cols, num_cycles, num_cycle_pts), dtype=np.float16)
        expected_2d = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
        for row_ind in range(num_rows):
            for col_ind in range(num_cols):
                for cycle_ind in range(num_cycles):
                    for bias_ind in range(num_cycle_pts):
                        val = 1E+3 * row_ind + 1E+2 * col_ind + 1E+1 * cycle_ind + bias_ind
                        expected_2d[row_ind * num_cols + col_ind, cycle_ind * num_cycle_pts + bias_ind] = val
                        source_nd[row_ind, col_ind, cycle_ind, bias_ind] = val

        # case 1: Pos and Spec both arranged as slow to fast:
        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles)))

        ret_2d, success = hdf_utils.reshape_from_n_dims(source_nd, h5_pos=source_pos_data, h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 2: Only Pos arranged as slow to fast:
        main_pos_sorted = np.transpose(source_nd, (0, 1, 3, 2))
        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T
        source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                      np.repeat(np.arange(num_cycles), num_cycle_pts),))

        ret_2d, success = hdf_utils.reshape_from_n_dims(main_pos_sorted, h5_pos=source_pos_data,
                                                        h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 3: only Spec arranged as slow to fast:
        main_spec_sorted = np.transpose(source_nd, (1, 0, 2, 3))
        source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                     np.repeat(np.arange(num_rows), num_cols))).T
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles)))

        ret_2d, success = hdf_utils.reshape_from_n_dims(main_spec_sorted, h5_pos=source_pos_data,
                                                        h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 4: neither pos nor spec arranged as slow to fast:
        main_not_sorted = np.transpose(source_nd, (1, 0, 3, 2))
        source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                     np.repeat(np.arange(num_rows), num_cols))).T
        source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                      np.repeat(np.arange(num_cycles), num_cycle_pts),))

        ret_2d, success = hdf_utils.reshape_from_n_dims(main_not_sorted, h5_pos=source_pos_data,
                                                        h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

    def test_reshape_from_n_dims_pos_and_spec_may_may_not_be_provided(self):
        num_rows = 3
        num_cols = 5
        num_cycles = 2
        num_cycle_pts = 7

        # the N dimensional dataset should be arranged in the following order:
        # [positions slowest to fastest, spectroscopic slowest to fastest]
        source_nd = np.zeros(shape=(num_rows, num_cols, num_cycles, num_cycle_pts), dtype=np.float16)
        expected_2d = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
        for row_ind in range(num_rows):
            for col_ind in range(num_cols):
                for cycle_ind in range(num_cycles):
                    for bias_ind in range(num_cycle_pts):
                        val = 1E+3 * row_ind + 1E+2 * col_ind + 1E+1 * cycle_ind + bias_ind
                        expected_2d[row_ind * num_cols + col_ind, cycle_ind * num_cycle_pts + bias_ind] = val
                        source_nd[row_ind, col_ind, cycle_ind, bias_ind] = val

        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles)))

        # case 1: only pos provided:
        ret_2d, success = hdf_utils.reshape_from_n_dims(source_nd, h5_pos=source_pos_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 2: only spec provided:
        ret_2d, success = hdf_utils.reshape_from_n_dims(source_nd, h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 3: neither pos nor spec provided:
        with self.assertRaises(ValueError):
            _ = hdf_utils.reshape_from_n_dims(source_nd)

    def test_get_all_main_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/results_main']]
            main_dsets = hdf_utils.get_all_main(h5_f, verbose=False)
            # self.assertEqual(set(main_dsets), set(expected_dsets))
            self.assertEqual(len(main_dsets), len(expected_dsets))
            self.assertTrue(np.all([x.name == y.name for x, y in zip(main_dsets, expected_dsets)]))

    def __validate_aux_dset_pair(self, h5_group, h5_inds, h5_vals, dim_names, dim_units, inds_matrix,
                                 vals_matrix=None, base_name=None, h5_main=None, is_spectral=True):
        if vals_matrix is None:
            vals_matrix = inds_matrix
        if base_name is None:
            if is_spectral:
                base_name = 'Spectroscopic'
            else:
                base_name = 'Position'
        else:
            self.assertIsInstance(base_name, (str, unicode))

        for h5_dset, exp_dtype, exp_name, ref_data in zip([h5_inds, h5_vals],
                                                          [write_utils.INDICES_DTYPE, write_utils.VALUES_DTYPE],
                                                          [base_name + '_Indices', base_name + '_Values'],
                                                          [inds_matrix, vals_matrix]):
            if isinstance(h5_main, h5py.Dataset):
                self.assertEqual(h5_main.file[h5_main.attrs[exp_name]], h5_dset)
            self.assertIsInstance(h5_dset, h5py.Dataset)
            self.assertEqual(h5_dset.parent, h5_group)
            self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
            self.assertTrue(np.allclose(ref_data, h5_dset[()]))
            self.assertEqual(h5_dset.dtype, exp_dtype)
            self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
            self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
            self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
            # assert region references
            for dim_ind, curr_name in enumerate(dim_names):
                expected = np.squeeze(ref_data[:, dim_ind])
                if is_spectral:
                    expected = np.squeeze(ref_data[dim_ind])
                self.assertTrue(np.allclose(expected,
                                            np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

    def test_write_ind_val_dsets_legal_bare_minimum_pos(self):
        num_cols = 3
        num_rows = 2
        sizes = [num_cols, num_rows]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        descriptor = []
        for length, name, units in zip(sizes, dim_names, dim_units):
            descriptor.append(write_utils.Dimension(name, units, np.arange(length)))

        pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols))).T
        file_path = 'test_write_ind_val_dsets.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_inds, h5_vals = hdf_utils.write_ind_val_dsets(h5_f, descriptor, is_spectral=False)

            self. __validate_aux_dset_pair(h5_f, h5_inds, h5_vals, dim_names, dim_units, pos_data,
                                           is_spectral=False)

        os.remove(file_path)

    def test_write_ind_val_dsets_legal_bare_minimum_spec(self):
        num_cols = 3
        num_rows = 2
        sizes = [num_cols, num_rows]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        descriptor = []
        for length, name, units in zip(sizes, dim_names, dim_units):
            descriptor.append(write_utils.Dimension(name, units, np.arange(length)))

        spec_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols)))
        file_path = 'test_write_ind_val_dsets.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group("Blah")
            h5_inds, h5_vals = hdf_utils.write_ind_val_dsets(h5_group, descriptor, is_spectral=True)

            self.__validate_aux_dset_pair(h5_group, h5_inds, h5_vals, dim_names, dim_units, spec_data,
                                          is_spectral=True)
        os.remove(file_path)

    def test_write_ind_val_dsets_legal_override_steps_offsets_base_name(self):
        num_cols = 2
        num_rows = 3
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']
        col_step = 0.25
        row_step = 0.05
        col_initial = 1
        row_initial = 0.2

        descriptor = []
        for length, name, units, step, initial in zip([num_cols, num_rows], dim_names, dim_units,
                                                      [col_step, row_step], [col_initial, row_initial]):
            descriptor.append(write_utils.Dimension(name, units, initial + step * np.arange(length)))

        new_base_name = 'Overriden'
        spec_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols)))
        spec_vals = np.vstack((np.tile(np.arange(num_cols), num_rows) * col_step + col_initial,
                              np.repeat(np.arange(num_rows), num_cols) * row_step + row_initial))

        file_path = 'test_write_ind_val_dsets.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group("Blah")
            h5_inds, h5_vals = hdf_utils.write_ind_val_dsets(h5_group, descriptor, is_spectral=True,
                                                             base_name=new_base_name)
            self.__validate_aux_dset_pair(h5_group, h5_inds, h5_vals, dim_names, dim_units, spec_inds,
                                          vals_matrix=spec_vals, base_name=new_base_name, is_spectral=True)
        os.remove(file_path)

    def test_write_ind_val_dsets_illegal(self):
        sizes = [3, 2]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        descriptor = []
        for length, name, units in zip(sizes, dim_names, dim_units):
            descriptor.append(write_utils.Dimension(name, units, np.arange(length)))

        file_path = 'test_write_ind_val_dsets.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            pass

        with self.assertRaises(ValueError):
            # h5_f should be valid in terms of type but closed
            _ = hdf_utils.write_ind_val_dsets(h5_f, descriptor)

        os.remove(file_path)

    def test_write_main_dset_small(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']

        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        spec_data = np.vstack((np.tile(np.arange(7), 2),
                              np.repeat(np.arange(2), 7)))

        with h5py.File(file_path) as h5_f:
            usid_main = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                      spec_dims, main_dset_attrs=None)
            self.assertIsInstance(usid_main, USIDataset)
            self.assertEqual(usid_main.name.split('/')[-1], main_data_name)
            self.assertEqual(usid_main.parent, h5_f)
            self.assertTrue(np.allclose(main_data, usid_main[()]))

            self.__validate_aux_dset_pair(h5_f, usid_main.h5_pos_inds, usid_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

            self.__validate_aux_dset_pair(h5_f, usid_main.h5_spec_inds, usid_main.h5_spec_vals, spec_names, spec_units,
                                          spec_data, h5_main=usid_main, is_spectral=True)
        os.remove(file_path)

    def test_write_main_dset_dask(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']

        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        spec_data = np.vstack((np.tile(np.arange(7), 2),
                              np.repeat(np.arange(2), 7)))

        with h5py.File(file_path) as h5_f:
            usid_main = hdf_utils.write_main_dataset(h5_f, da.from_array(main_data, chunks=main_data.shape),
                                                     main_data_name, quantity, dset_units, pos_dims,
                                                     spec_dims, main_dset_attrs=None)
            self.assertIsInstance(usid_main, USIDataset)
            self.assertEqual(usid_main.name.split('/')[-1], main_data_name)
            self.assertEqual(usid_main.parent, h5_f)
            self.assertTrue(np.allclose(main_data, usid_main[()]))

            self.__validate_aux_dset_pair(h5_f, usid_main.h5_pos_inds, usid_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

            self.__validate_aux_dset_pair(h5_f, usid_main.h5_spec_inds, usid_main.h5_spec_vals, spec_names, spec_units,
                                          spec_data, h5_main=usid_main, is_spectral=True)
        os.remove(file_path)

    def test_write_main_dset_empty(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        main_data = (15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']

        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        spec_data = np.vstack((np.tile(np.arange(7), 2),
                              np.repeat(np.arange(2), 7)))

        with h5py.File(file_path) as h5_f:
            usid_main = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                      spec_dims, dtype=np.float16, main_dset_attrs=None)
            self.assertIsInstance(usid_main, USIDataset)
            self.assertEqual(usid_main.name.split('/')[-1], main_data_name)
            self.assertEqual(usid_main.parent, h5_f)
            self.assertEqual(main_data, usid_main.shape)

            self.__validate_aux_dset_pair(h5_f, usid_main.h5_pos_inds, usid_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

            self.__validate_aux_dset_pair(h5_f, usid_main.h5_spec_inds, usid_main.h5_spec_vals, spec_names, spec_units,
                                          spec_data, h5_main=usid_main, is_spectral=True)
        os.remove(file_path)

    def test_write_main_existing_spec_aux(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']
        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        spec_data = np.vstack((np.tile(np.arange(7), 2),
                               np.repeat(np.arange(2), 7)))

        with h5py.File(file_path) as h5_f:
            h5_spec_inds, h5_spec_vals = hdf_utils.write_ind_val_dsets(h5_f, spec_dims, is_spectral=True)
            self.__validate_aux_dset_pair(h5_f, h5_spec_inds, h5_spec_vals, spec_names, spec_units, spec_data,
                                          is_spectral=True)

            usid_main = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                      None, h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals,
                                                      main_dset_attrs=None)

            self.__validate_aux_dset_pair(h5_f, usid_main.h5_pos_inds, usid_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

        os.remove(file_path)

    def test_write_main_existing_both_aux(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']
        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        spec_data = np.vstack((np.tile(np.arange(7), 2),
                               np.repeat(np.arange(2), 7)))

        with h5py.File(file_path) as h5_f:
            h5_spec_inds, h5_spec_vals = hdf_utils.write_ind_val_dsets(h5_f, spec_dims, is_spectral=True)
            h5_pos_inds, h5_pos_vals = hdf_utils.write_ind_val_dsets(h5_f, pos_dims, is_spectral=False)

            usid_main = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, None,
                                                      None, h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals,
                                                      h5_pos_vals=h5_pos_vals, h5_pos_inds=h5_pos_inds,
                                                      main_dset_attrs=None)

            self.__validate_aux_dset_pair(h5_f, h5_pos_inds, h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

            self.__validate_aux_dset_pair(h5_f, h5_spec_inds, h5_spec_vals, spec_names,spec_units,
                                          spec_data, h5_main=usid_main, is_spectral=True)
        os.remove(file_path)

    def test_write_main_dset_prod_sizes_mismatch(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 15]  # too many steps in the Y direction
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']
        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))

        with h5py.File(file_path) as h5_f:
            with self.assertRaises(ValueError):
                _ = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                 spec_dims)
        os.remove(file_path)

    def test_write_reduced_anc_dsets_spec_2d_to_1d(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)
        shutil.copy(data_utils.std_beps_path, duplicate_path)
        with h5py.File(duplicate_path) as h5_f:
            h5_spec_inds_orig = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_spec_vals_orig = h5_f['/Raw_Measurement/Spectroscopic_Values']
            new_base_name = 'Blah'
            # cycle_starts = np.where(h5_spec_inds_orig[0] == 0)[0]
            h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_anc_dsets(h5_spec_inds_orig.parent,
                                                                                   h5_spec_inds_orig,
                                                                                   h5_spec_vals_orig,
                                                                                   'Bias',
                                                                                   basename=new_base_name)

            dim_names = ['Cycle']
            dim_units = ['']
            ref_data = np.expand_dims(np.arange(2), axis=0)
            for h5_dset, exp_dtype, exp_name in zip([h5_spec_inds_new, h5_spec_vals_new],
                                                    [h5_spec_inds_orig.dtype, h5_spec_vals_orig.dtype],
                                                    [new_base_name + '_Indices', new_base_name + '_Values']):
                self.assertIsInstance(h5_dset, h5py.Dataset)
                self.assertEqual(h5_dset.parent, h5_spec_inds_orig.parent)
                self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
                self.assertTrue(np.allclose(ref_data, h5_dset[()]))
                self.assertEqual(h5_dset.dtype, exp_dtype)
                self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
                # assert region references
                for dim_ind, curr_name in enumerate(dim_names):
                    self.assertTrue(np.allclose(np.squeeze(ref_data[dim_ind]),
                                                np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

        os.remove(duplicate_path)

    def test_write_reduced_anc_dsets_spec_1d_to_0d(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path) as h5_f:

            h5_spec_inds_orig, h5_spec_vals_orig = hdf_utils.write_ind_val_dsets(h5_f,
                                                                                 write_utils.Dimension('Bias', 'V', 10),
                                                                                 is_spectral=True)
            new_base_name = 'Blah'
            h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_anc_dsets(h5_f, h5_spec_inds_orig,
                                                                                   h5_spec_vals_orig,
                                                                                   'Bias', basename=new_base_name)

            dim_names = ['Single_Step']
            dim_units = ['a. u.']
            ref_data = np.expand_dims(np.arange(1), axis=0)
            for h5_dset, exp_dtype, exp_name in zip([h5_spec_inds_new, h5_spec_vals_new],
                                                    [h5_spec_inds_orig.dtype, h5_spec_vals_orig.dtype],
                                                    [new_base_name + '_Indices', new_base_name + '_Values']):
                self.assertIsInstance(h5_dset, h5py.Dataset)
                self.assertEqual(h5_dset.parent, h5_spec_inds_orig.parent)
                self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
                self.assertTrue(np.allclose(ref_data, h5_dset[()]))
                self.assertEqual(h5_dset.dtype, exp_dtype)
                self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
                # assert region references
                for dim_ind, curr_name in enumerate(dim_names):
                    self.assertTrue(np.allclose(np.squeeze(ref_data[dim_ind]),
                                                np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

        os.remove(duplicate_path)

    def test_write_reduced_anc_dsets_3d_to_1d_pos_fastest_n_slowest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path) as h5_f:

            dims = [write_utils.Dimension('X', 'nm', np.linspace(300, 350, 5)),
                    write_utils.Dimension('Y', 'um', [-2, 4, 10]),
                    write_utils.Dimension('Z', 'm', 2)]

            h5_spec_inds_orig, h5_spec_vals_orig = hdf_utils.write_ind_val_dsets(h5_f, dims, is_spectral=False)
            new_base_name = 'Position'
            h5_grp = h5_f.create_group('My_Group')
            h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_anc_dsets(h5_grp, h5_spec_inds_orig,
                                                                                   h5_spec_vals_orig, ['X', 'Z'])

            dim_names = ['Y']
            dim_units = ['um']
            ref_inds = np.expand_dims(np.arange(3), axis=1)
            ref_vals = np.expand_dims([-2, 4, 10], axis=1)
            for h5_dset, exp_dtype, exp_name, ref_data in zip([h5_spec_inds_new, h5_spec_vals_new],
                                                              [h5_spec_inds_orig.dtype, h5_spec_vals_orig.dtype],
                                                              [new_base_name + '_Indices', new_base_name + '_Values'],
                                                              [ref_inds, ref_vals]):
                self.assertIsInstance(h5_dset, h5py.Dataset)
                self.assertEqual(h5_dset.parent, h5_grp)
                self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
                self.assertTrue(np.allclose(ref_data, h5_dset[()]))
                self.assertEqual(h5_dset.dtype, exp_dtype)
                self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
                # assert region references
                for dim_ind, curr_name in enumerate(dim_names):
                    self.assertTrue(np.allclose(np.squeeze(ref_data[:, dim_ind]),
                                                np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

        os.remove(duplicate_path)

    def test_write_reduced_anc_dsets_3d_to_1d_spec_fastest_n_slowest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path) as h5_f:

            dims = [write_utils.Dimension('Freq', 'Hz', np.linspace(300, 350, 5)),
                    write_utils.Dimension('Bias', 'V', [-2, 4, 10]),
                    write_utils.Dimension('Cycle', 'a.u.', 2)]

            h5_spec_inds_orig, h5_spec_vals_orig = hdf_utils.write_ind_val_dsets(h5_f, dims, is_spectral=True)
            new_base_name = 'Blah'
            h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_anc_dsets(h5_f, h5_spec_inds_orig,
                                                                                   h5_spec_vals_orig,
                                                                                   ['Freq', 'Cycle'],
                                                                                   basename=new_base_name)

            dim_names = ['Bias']
            dim_units = ['V']
            ref_inds = np.expand_dims(np.arange(3), axis=0)
            ref_vals = np.expand_dims([-2, 4, 10], axis=0)
            for h5_dset, exp_dtype, exp_name, ref_data in zip([h5_spec_inds_new, h5_spec_vals_new],
                                                              [h5_spec_inds_orig.dtype, h5_spec_vals_orig.dtype],
                                                              [new_base_name + '_Indices', new_base_name + '_Values'],
                                                              [ref_inds, ref_vals]):
                self.assertIsInstance(h5_dset, h5py.Dataset)
                self.assertEqual(h5_dset.parent, h5_spec_inds_orig.parent)
                self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
                self.assertTrue(np.allclose(ref_data, h5_dset[()]))
                self.assertEqual(h5_dset.dtype, exp_dtype)
                self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
                # assert region references
                for dim_ind, curr_name in enumerate(dim_names):
                    self.assertTrue(np.allclose(np.squeeze(ref_data[dim_ind]),
                                                np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

        os.remove(duplicate_path)

    def test_write_reduced_anc_dsets_3d_to_1d_spec_fastest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path) as h5_f:

            dims = [write_utils.Dimension('Freq', 'Hz', np.linspace(300, 350, 5)),
                    write_utils.Dimension('Bias', 'V', [-2, 4, 10]),
                    write_utils.Dimension('Cycle', 'a.u.', 2)]

            h5_spec_inds_orig, h5_spec_vals_orig = hdf_utils.write_ind_val_dsets(h5_f, dims, is_spectral=True)
            new_base_name = 'Blah'
            h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_anc_dsets(h5_f, h5_spec_inds_orig,
                                                                                   h5_spec_vals_orig,
                                                                                   ['Freq', 'Bias'],
                                                                                   basename=new_base_name)

            dim_names = ['Cycle']
            dim_units = ['a.u.']
            ref_inds = np.expand_dims(np.arange(2), axis=0)
            ref_vals = np.expand_dims([0, 1], axis=0)
            for h5_dset, exp_dtype, exp_name, ref_data in zip([h5_spec_inds_new, h5_spec_vals_new],
                                                              [h5_spec_inds_orig.dtype, h5_spec_vals_orig.dtype],
                                                              [new_base_name + '_Indices', new_base_name + '_Values'],
                                                              [ref_inds, ref_vals]):
                self.assertIsInstance(h5_dset, h5py.Dataset)
                self.assertEqual(h5_dset.parent, h5_spec_inds_orig.parent)
                self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
                self.assertTrue(np.allclose(ref_data, h5_dset[()]))
                self.assertEqual(h5_dset.dtype, exp_dtype)
                self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
                # assert region references
                for dim_ind, curr_name in enumerate(dim_names):
                    self.assertTrue(np.allclose(np.squeeze(ref_data[dim_ind]),
                                                np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

        os.remove(duplicate_path)

    def test_write_reduced_anc_dsets_3d_to_1d_spec_slowest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path) as h5_f:

            dims = [write_utils.Dimension('Freq', 'Hz', np.linspace(300, 350, 5)),
                    write_utils.Dimension('Bias', 'V', [-2, 4, 10]),
                    write_utils.Dimension('Cycle', 'a.u.', 2)]

            h5_spec_inds_orig, h5_spec_vals_orig = hdf_utils.write_ind_val_dsets(h5_f, dims, is_spectral=True)
            new_base_name = 'Blah'
            h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_anc_dsets(h5_f, h5_spec_inds_orig,
                                                                                   h5_spec_vals_orig,
                                                                                   ['Cycle', 'Bias'],
                                                                                   basename=new_base_name)

            dim_names = ['Freq']
            dim_units = ['Hz']
            ref_inds = np.expand_dims(np.arange(5), axis=0)
            ref_vals = np.expand_dims(np.linspace(300, 350, 5), axis=0)
            for h5_dset, exp_dtype, exp_name, ref_data in zip([h5_spec_inds_new, h5_spec_vals_new],
                                                              [h5_spec_inds_orig.dtype, h5_spec_vals_orig.dtype],
                                                              [new_base_name + '_Indices', new_base_name + '_Values'],
                                                              [ref_inds, ref_vals]):
                self.assertIsInstance(h5_dset, h5py.Dataset)
                self.assertEqual(h5_dset.parent, h5_spec_inds_orig.parent)
                self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
                self.assertTrue(np.allclose(ref_data, h5_dset[()]))
                self.assertEqual(h5_dset.dtype, exp_dtype)
                self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
                # assert region references
                for dim_ind, curr_name in enumerate(dim_names):
                    self.assertTrue(np.allclose(np.squeeze(ref_data[dim_ind]),
                                                np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

        os.remove(duplicate_path)

    def test_write_reduced_anc_dsets_3d_to_2d_spec_fastest_n_slowest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path) as h5_f:

            dims = [write_utils.Dimension('Freq', 'Hz', np.linspace(300, 350, 5)),
                    write_utils.Dimension('Bias', 'V', [-2, 4, 10]),
                    write_utils.Dimension('Cycle', 'a.u.', 2)]

            h5_spec_inds_orig, h5_spec_vals_orig = hdf_utils.write_ind_val_dsets(h5_f, dims, is_spectral=True)
            new_base_name = 'Blah'
            h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_anc_dsets(h5_f, h5_spec_inds_orig,
                                                                                   h5_spec_vals_orig,
                                                                                   ['Bias'],
                                                                                   basename=new_base_name)

            dim_names = ['Freq', 'Cycle']
            dim_units = ['Hz', 'a.u.']
            ref_vals = np.vstack((np.tile(np.linspace(300, 350, 5), 2),
                                  np.repeat(np.arange(2), 5)))
            ref_inds = np.vstack((np.tile(np.arange(5, dtype=np.uint16), 2),
                                  np.repeat(np.arange(2, dtype=np.uint16), 5)))
            for h5_dset, exp_dtype, exp_name, ref_data in zip([h5_spec_inds_new, h5_spec_vals_new],
                                                              [h5_spec_inds_orig.dtype, h5_spec_vals_orig.dtype],
                                                              [new_base_name + '_Indices', new_base_name + '_Values'],
                                                              [ref_inds, ref_vals]):
                self.assertIsInstance(h5_dset, h5py.Dataset)
                self.assertEqual(h5_dset.parent, h5_spec_inds_orig.parent)
                self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
                self.assertTrue(np.allclose(ref_data, h5_dset[()]))
                self.assertEqual(h5_dset.dtype, exp_dtype)
                self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
                # assert region references
                for dim_ind, curr_name in enumerate(dim_names):
                    self.assertTrue(np.allclose(np.squeeze(ref_data[dim_ind]),
                                                np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

        os.remove(duplicate_path)


if __name__ == '__main__':
    unittest.main()
