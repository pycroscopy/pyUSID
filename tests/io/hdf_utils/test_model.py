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

from tests.io import data_utils


if sys.version_info.major == 3:
    unicode = str


class TestModel(unittest.TestCase):

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


class TestGetDimensionality(TestModel):

    def test_legal_no_sort(self):
        self.__helper_no_sort(hdf_dsets=True)
        self.__helper_no_sort(hdf_dsets=False)

    def __helper_no_sort(self, hdf_dsets=True):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_shapes = [[7, 2],
                               [7],
                               [5, 3]]
            for h5_dset, exp_shape in zip(h5_dsets, expected_shapes):
                if not hdf_dsets:
                    h5_dset = h5_dset[()]
                self.assertTrue(np.all(exp_shape == hdf_utils.get_dimensionality(h5_dset)))

    def test_legal_w_sort(self):
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

    def test_not_hdf_dset(self):
        for obj in [15, 'srds']:
            with self.assertRaises(TypeError):
                _ = hdf_utils.get_dimensionality(obj)

    def test_invalid_sort(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_dset = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            with self.assertRaises(ValueError):
                _ = hdf_utils.get_dimensionality(h5_dset, index_sort=[3, 4])
                _ = hdf_utils.get_dimensionality(h5_dset, index_sort=['a', np.arange(5)])


class TestGetSortOrder(TestModel):

    def test_invalid_types(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            for obj in ['fdfdfd', h5_f]:
                with self.assertRaises(TypeError):
                    _ = hdf_utils.get_sort_order(obj)

    def test_simple(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_order = [[0, 1], [0], [0, 1]]
            for h5_dset, exp_order in zip(h5_dsets, expected_order):
                self.assertTrue(np.all(exp_order == hdf_utils.get_sort_order(h5_dset)))

    def test_reversed(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_dsets = [np.flipud(h5_f['/Raw_Measurement/Spectroscopic_Indices']),
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        np.fliplr(h5_f['/Raw_Measurement/Position_Indices'])]
            expected_order = [[1, 0], [0], [1, 0]]
            for h5_dset, exp_order in zip(h5_dsets, expected_order):
                self.assertTrue(np.all(exp_order == hdf_utils.get_sort_order(h5_dset)))


class TestGetUnitValues(TestModel):

    def test_source_spec_all(self):
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

    def test_source_spec_all_explicit(self):
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

    def test_illegal_key(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Does not exist'])

    def test_illegal_dset(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Ancillary']
            with self.assertRaises(ValueError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Bias'])

    def test_source_spec_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float32(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)))}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names='Bias')
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_source_pos_all(self):
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

    def test_source_pos_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_vals = h5_f['/Raw_Measurement/Position_Values']
            num_rows = 3
            expected = {'Y': np.float16(np.arange(num_rows) * 1.25)}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names='Y')
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_all_dim_names_not_provided(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices'][()]
            h5_vals = h5_f['/Raw_Measurement/Position_Values'][()]

            with self.assertRaises(TypeError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Y'])

    def test_dependent_dim(self):
        with h5py.File(data_utils.relaxation_path, mode='r') as h5_f:
            h5_inds = h5_f['/Measurement_000/Channel_000/Spectroscopic_Indices']
            h5_vals = h5_f['/Measurement_000/Channel_000/Spectroscopic_Values']
            spec_dim_names = hdf_utils.get_attr(h5_inds, 'labels')
            ret_dict = hdf_utils.get_unit_values(h5_inds, h5_vals)
            for dim_ind, dim_name in enumerate(spec_dim_names):
                exp_val = hdf_utils.get_attr(h5_inds, 'unit_vals_dim_' + str(dim_ind))
                act_val = ret_dict[dim_name]
                self.assertTrue(np.allclose(exp_val, act_val))

    def test_sparse_samp_no_attr(self):
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

    def test_sparse_samp_w_attr(self):
        # What should the user expect this function to do? throw an error.
        with h5py.File(data_utils.sparse_sampling_path, mode='r') as h5_f:
            h5_inds = h5_f['/Measurement_000/Channel_001/Position_Indices']
            h5_vals = h5_f['/Measurement_000/Channel_001/Position_Values']

            with self.assertRaises(ValueError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Y'])

    def test_incomp_dim_no_attr(self):
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


class TestReshapeToNDims(TestModel):

    def test_h5_no_sort_reqd(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main, get_labels=True, sort_dims=False,
                                                                 lazy=False)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['X', 'Y', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(h5_main[()], (num_rows, num_cols, num_cycles, num_cycle_pts))
            expected_n_dim = np.transpose(expected_n_dim, (1, 0, 3, 2))
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main, get_labels=True, sort_dims=True,
                                                                 lazy=False)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['X', 'Y', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(h5_main[()], (num_rows, num_cols, num_cycles, num_cycle_pts))
            expected_n_dim = np.transpose(expected_n_dim, (1, 0, 3, 2))
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

    def test_h5_not_main_dset(self):
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

    def test_numpy(self):
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
                                                     h5_spec=source_spec_data, get_labels=False, lazy=False)
        expected_n_dim = np.reshape(source_main_data, (num_rows, num_cols, num_cycles, num_cycle_pts))
        self.assertTrue(np.allclose(expected_n_dim, n_dim))

    def test_sort_required(self):
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
                                                                 lazy=False)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['Y', 'X', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(source_main_data, (num_rows, num_cols, num_cycles, num_cycle_pts))
            expected_n_dim = np.transpose(expected_n_dim, [1, 0, 3, 2])
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

        os.remove(file_path)


class TestReshapeFromNDims(TestModel):

    def test_pos_and_spec_provided(self):
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

    def test_pos_and_spec_may_may_not_be_provided(self):
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


class TestWriteMainDataset(TestModel):

    def test_small(self):
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

            data_utils.validate_aux_dset_pair(self, h5_f, usid_main.h5_pos_inds, usid_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

            data_utils.validate_aux_dset_pair(self, h5_f, usid_main.h5_spec_inds, usid_main.h5_spec_vals, spec_names, spec_units,
                                          spec_data, h5_main=usid_main, is_spectral=True)
        os.remove(file_path)

    def test_dask(self):
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

            data_utils.validate_aux_dset_pair(self, h5_f, usid_main.h5_pos_inds, usid_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

            data_utils.validate_aux_dset_pair(self, h5_f, usid_main.h5_spec_inds, usid_main.h5_spec_vals, spec_names, spec_units,
                                          spec_data, h5_main=usid_main, is_spectral=True)
        os.remove(file_path)

    def test_empty(self):
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

            data_utils.validate_aux_dset_pair(self, h5_f, usid_main.h5_pos_inds, usid_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

            data_utils.validate_aux_dset_pair(self, h5_f, usid_main.h5_spec_inds, usid_main.h5_spec_vals, spec_names, spec_units,
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
            data_utils.validate_aux_dset_pair(self,h5_f, h5_spec_inds, h5_spec_vals, spec_names, spec_units, spec_data,
                                          is_spectral=True)

            usid_main = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                      None, h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals,
                                                      main_dset_attrs=None)

            data_utils.validate_aux_dset_pair(self,h5_f, usid_main.h5_pos_inds, usid_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

        os.remove(file_path)

    def test_existing_both_aux(self):
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

            data_utils.validate_aux_dset_pair(self,h5_f, h5_pos_inds, h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False)

            data_utils.validate_aux_dset_pair(self,h5_f, h5_spec_inds, h5_spec_vals, spec_names,spec_units,
                                          spec_data, h5_main=usid_main, is_spectral=True)
        os.remove(file_path)

    def test_prod_sizes_mismatch(self):
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


if __name__ == '__main__':
    unittest.main()
