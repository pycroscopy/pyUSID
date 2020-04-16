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
            expected = {}
            for dim_name in ['Bias', 'Cycle']:
                expected[dim_name] = h5_f['/Raw_Measurement/' + dim_name][()]
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals)
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_source_spec_all_explicit(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            expected = {}
            for dim_name in ['Bias', 'Cycle']:
                expected[dim_name] = h5_f['/Raw_Measurement/' + dim_name][()]
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
            expected = {'Bias': h5_f['/Raw_Measurement/Bias'][()]}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names='Bias')
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_source_pos_all(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_vals = h5_f['/Raw_Measurement/Position_Values']
            expected = {}
            for dim_name in ['X', 'Y']:
                expected[dim_name] = h5_f['/Raw_Measurement/' + dim_name][()]
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals)
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_source_pos_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_vals = h5_f['/Raw_Measurement/Position_Values']
            expected = {'Y': h5_f['/Raw_Measurement/Y'][()]}
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

    def test_h5_already_sorted(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            nd_slow_to_fast = h5_f['/Raw_Measurement/n_dim_form'][()]

            h5_main = h5_f['/Raw_Measurement/source_main']
            # Data is always slowest to fastest
            # Anc dims arranged from fastest to slowest
            # Expecting data dims to be arranged according to anc dims order
            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main, get_labels=True, sort_dims=False,
                                                                 lazy=False, verbose=True)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['X', 'Y', 'Bias', 'Cycle'])]))
            self.assertTrue(success)
            nd_fast_to_slow = nd_slow_to_fast.transpose(1, 0, 3, 2)
            self.assertTrue(np.allclose(nd_fast_to_slow, n_dim))

            # Anc dims arranged from fastest to slowest
            # Expecting data dims to be arranged according to slow to fast
            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main, get_labels=True, sort_dims=True,
                                                                 lazy=False, verbose=True)
            self.assertTrue(success)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['Y', 'X', 'Cycle', 'Bias'])]))
            self.assertTrue(np.allclose(nd_slow_to_fast, n_dim))

    def test_h5_manually_provided_anc_dsets_h5(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            nd_slow_to_fast = h5_f['/Raw_Measurement/n_dim_form'][()]
            nd_fast_to_slow = nd_slow_to_fast.transpose(1, 0, 3, 2)
            exp_labs = ['X', 'Y', 'Bias', 'Cycle']


            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_spec_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']

            # BOTH POS AND SPEC
            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main,
                                                                 h5_pos=h5_pos_inds,
                                                                 h5_spec=h5_spec_inds,
                                                                 get_labels=True,
                                                                 sort_dims=False,
                                                                 lazy=False, verbose=True)
            self.assertTrue(np.all([x == y for x, y in zip(labels, exp_labs)]))
            self.assertTrue(success)
            self.assertTrue(np.allclose(nd_fast_to_slow, n_dim))

            # ONLY POS:
            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main,
                                                                 h5_pos=h5_pos_inds,
                                                                 h5_spec=None,
                                                                 get_labels=True,
                                                                 sort_dims=False,
                                                                 lazy=False,
                                                                 verbose=True)
            self.assertTrue(np.all([x == y for x, y in zip(labels, exp_labs)]))
            self.assertTrue(success)
            self.assertTrue(np.allclose(nd_fast_to_slow, n_dim))

            # ONLY SPEC
            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main,
                                                                 h5_pos=None,
                                                                 h5_spec=h5_spec_inds,
                                                                 get_labels=True,
                                                                 sort_dims=False,
                                                                 lazy=False,
                                                                 verbose=True)
            self.assertTrue(np.all([x == y for x, y in zip(labels, exp_labs)]))
            self.assertTrue(success)
            self.assertTrue(np.allclose(nd_fast_to_slow, n_dim))

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

    def build_main_anc_4d(self):
        num_rows = 3
        num_cols = 5
        num_cycles = 2
        num_cycle_pts = 7
        # arrange as fast, slow
        pos_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols))).T
        # arrange as fast, slow
        spec_inds = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                               np.repeat(np.arange(num_cycles), num_cycle_pts)))

        # Data is arranged from slowest to fastest
        main_nd = np.zeros(shape=(num_rows, num_cols, num_cycles,
                                  num_cycle_pts), dtype=np.uint8)
        for row_ind in range(num_rows):
            for col_ind in range(num_cols):
                for cycle_ind in range(num_cycles):
                    # for bias_ind in range(num_cycle_pts):
                    val = 1E+3*row_ind + 1E+2*col_ind + 1E+1*cycle_ind + np.arange(num_cycle_pts)
                    main_nd[row_ind, col_ind, cycle_ind] = val

        return main_nd, pos_inds, spec_inds

    def base_comparison_4d(self, flip_pos_inds, flip_spec_inds, lazy_in=False,
                           lazy_out=False, verbose=False):
        # Generated Data dims from slowest to fastest
        exp_nd_s2f, pos_inds, spec_inds = self.build_main_anc_4d()
        # nd    (Y, X, Cycle, Bias)
        main_2d = exp_nd_s2f.reshape(np.prod(exp_nd_s2f.shape[:2]),
                                  np.prod(exp_nd_s2f.shape[2:]))

        # Dimension names arranged from slowest to fastest
        labs_s2f = ['Position Dimension 1', 'Position Dimension 0',
                    'Spectral Dimension 1', 'Spectral Dimension 0']

        # Generated ancillary dimensions are arranged from fastest to slowest
        # Unless any flipping is requested, as-is order should be fast to slow
        as_is_nd_order = [1, 0, 3, 2]
        # Unless any flipping is requested, s2f order is already in place
        s2f_lab_order = [0, 1, 2, 3]
        if flip_pos_inds:
            # arranged as slow to fast
            pos_inds = np.fliplr(pos_inds)
            as_is_nd_order = as_is_nd_order[:2][::-1] + as_is_nd_order[2:]
            s2f_lab_order = [1, 0] + s2f_lab_order[2:]
        if flip_spec_inds:
            # arranged as slow to fast
            as_is_nd_order = as_is_nd_order[:2] + as_is_nd_order[2:][::-1]
            s2f_lab_order = s2f_lab_order[:2] + [3, 2]
            spec_inds = np.flipud(spec_inds)

        if lazy_in:
            main_2d = da.from_array(main_2d, chunks=main_2d.shape)
            pos_inds = da.from_array(pos_inds, chunks=pos_inds.shape)
            spec_inds = da.from_array(spec_inds, chunks=spec_inds.shape)

        n_dim, suc, labs = hdf_utils.reshape_to_n_dims(main_2d,
                                                       h5_pos=pos_inds,
                                                     h5_spec=spec_inds, sort_dims=True,
                                                       get_labels=True,
                                                       lazy=lazy_out,
                                                       verbose=verbose)
        if lazy_out:
            self.assertIsInstance(n_dim, da.core.Array)
        self.assertTrue(np.allclose(exp_nd_s2f, n_dim))
        self.assertTrue(suc)
        # labels were auto-generated and these will be flipped blindly
        exp_labs = np.array(labs_s2f)[s2f_lab_order]
        self.assertTrue(np.all([x == y for x, y in zip(labs, exp_labs)]))

        if verbose:
            print('~~~~~~~~~~~~~~~~~~~~~~ UNSORTED ~~~~~~~~~~~~~~~~~~~~~~~~~')

        n_dim, suc, labs = hdf_utils.reshape_to_n_dims(main_2d,
                                                       h5_pos=pos_inds,
                                                     h5_spec=spec_inds,
                                                     sort_dims=False,
                                                       get_labels=True,
                                                       lazy=lazy_out,
                                                       verbose=verbose)
        if lazy_out:
            self.assertIsInstance(n_dim, da.core.Array)

        # Rearrange the dim labels and N-dim form from slow-to-fast to:
        if verbose:
            print('N-dim order will be permuted as: {}'.format(as_is_nd_order))
            print('Labels will be permuted as: {}'.format([1, 0, 3, 2]))
        exp_nd = exp_nd_s2f.transpose(tuple(as_is_nd_order))
        """
        This is sort of confusing:
        No matter how the pos / spec dims are ordered, the names will always
        start as P0, P1, S0, S1
        """
        exp_labs = np.array(labs_s2f)[[1, 0, 3, 2]]
        if verbose:
            print('Expected N-dim shape: {} and labels: {}'
                  ''.format(exp_nd.shape, exp_labs))

        self.assertTrue(np.allclose(exp_nd,  n_dim))
        self.assertTrue(suc)
        self.assertTrue(np.all([x == y for x, y in zip(labs, exp_labs)]))

    def test_numpy_ordinary(self):
        self.base_comparison_4d(False, False)

    def test_dask_input(self):
        self.base_comparison_4d(False, False, lazy_in=True, lazy_out=False)

    def test_dask_output(self):
        self.base_comparison_4d(False, False, lazy_in=False, lazy_out=True)

    def test_dask_all(self):
        self.base_comparison_4d(False, False, lazy_in=True, lazy_out=True)

    def test_numpy_pos_inds_order_flipped(self):
        self.base_comparison_4d(True, False)

    def test_numpy_spec_inds_order_flipped(self):
        # This is the same situation as in BEPS
        self.base_comparison_4d(False, True)

    def test_numpy_both_inds_order_flipped(self):
        self.base_comparison_4d(True, True)

    def test_dask_all_both_inds_order_flipped(self):
        self.base_comparison_4d(True, True, lazy_in=True, lazy_out=True)

    def build_main_anc_1_2d(self, is_2d=True, is_spec=False):
        num_rows = 2
        num_cols = 3
        # arrange as fast, slow
        pos_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols))).T

        # Data is arranged from slowest to fastest
        main_nd = np.random.randint(0, high=255, size=(num_rows, num_cols),
                                    dtype=np.uint8)
        if not is_2d:
            pos_inds = np.expand_dims(np.arange(num_rows), axis=1)
            main_nd = np.random.randint(0, high=255, size=num_rows,
                                        dtype=np.uint8)

        spec_inds= np.expand_dims([0], axis=0)

        if is_spec:
            return main_nd, spec_inds, pos_inds.T

        return main_nd, pos_inds, spec_inds

    def base_comparison_1_2d(self, is_2d, is_spec, flip_inds,
                             lazy_in=False, lazy_out=False):
        # Data is always stored from fastest to slowest
        # By default the ancillary dimensions are arranged from fastest to slowest
        main_nd, pos_inds, spec_inds = self.build_main_anc_1_2d(is_2d=is_2d,
                                                                is_spec=is_spec)

        main_2d = main_nd.reshape(-1, 1)
        main_nd_w_sing = np.expand_dims(main_nd, axis=-1)
        if is_spec:
            main_2d = main_2d.T
            main_nd_w_sing = np.expand_dims(main_nd, axis=0)

            # nd    (Y, X)
        order = [1, 0, 2]
        if is_spec:
            order = [0, 2, 1]
        if flip_inds:
            # arranged as slow to fast
            if is_spec:
                spec_inds = np.flipud(spec_inds)
                order = [0] + order[1:][::-1]
            else:
                pos_inds = np.fliplr(pos_inds)
                order = order[:2][::-1] + [2]

        print('2D: {}, Spec: {}, Flip: {}'.format(is_2d, is_spec, flip_inds))
        print('Main data shapes ND: {}, 2D: {}'.format(main_nd.shape, main_2d.shape))

        print(main_nd)
        print(main_2d)

        if lazy_in:
            main_2d = da.from_array(main_2d, chunks=main_2d.shape)

        n_dim, success = hdf_utils.reshape_to_n_dims(main_2d, h5_pos=pos_inds,
                                                     h5_spec=spec_inds,
                                                     sort_dims=True,
                                                     get_labels=False,
                                                     lazy=lazy_out,
                                                     verbose=True)
        if lazy_out:
            self.assertIsInstance(n_dim, da.core.Array)
        self.assertTrue(np.allclose(main_nd_w_sing, n_dim))

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        n_dim, success = hdf_utils.reshape_to_n_dims(main_2d, h5_pos=pos_inds,
                                                     h5_spec=spec_inds,
                                                     sort_dims=False,
                                                     get_labels=False,
                                                     lazy=lazy_out,
                                                     verbose=True)
        if lazy_out:
            self.assertIsInstance(n_dim, da.core.Array)

        if is_2d:
            main_nd_w_sing = main_nd_w_sing.transpose(order)

        self.assertTrue(np.allclose(main_nd_w_sing,  n_dim))

    def test_numpy_ordinary_1d_pos(self):
        self.base_comparison_1_2d(False, False, False)

    def test_dask_in_ordinary_1d_pos(self):
        self.base_comparison_1_2d(False, False, False,
                                  lazy_in=True, lazy_out=False)

    def test_dask_out_ordinary_1d_pos(self):
        self.base_comparison_1_2d(False, False, False,
                                  lazy_in=False, lazy_out=True)

    def test_dask_all_ordinary_1d_pos(self):
        self.base_comparison_1_2d(False, False, False,
                                  lazy_in=True, lazy_out=True)

    def test_numpy_ordinary_1d_spec(self):
        self.base_comparison_1_2d(False, True, False)

    def test_dask_in_ordinary_1d_spec(self):
        self.base_comparison_1_2d(False, True, False,
                                  lazy_in=True, lazy_out=False)

    def test_dask_out_ordinary_1d_spec(self):
        self.base_comparison_1_2d(False, True, False,
                                  lazy_in=False, lazy_out=True)

    def test_dask_all_ordinary_1d_spec(self):
        self.base_comparison_1_2d(False, True, False,
                                  lazy_in=True, lazy_out=True)

    def test_numpy_ordinary_2d_pos(self):
        self.base_comparison_1_2d(True, False, False)

    def test_numpy_ordinary_2d_spec(self):
        self.base_comparison_1_2d(True, True, False)



    def test_h5_both_inds_flipped(self):
        # Flipping both the spec and pos dimensions means that the order in which
        # the data is stored is the same order in which dimensions are arranged
        # In other words, sort should make no difference at all!
        file_path = 'reshape_to_n_dim_sort_required.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_raw_grp = h5_f.create_group('Raw_Measurement')

            main_nd, source_pos_data, source_spec_data = self.build_main_anc_4d()

            # arrange as slow, fast instead of fast, slow
            source_pos_data = np.fliplr(source_pos_data)
            # make spectroscopic slow, fast instead of fast, slow
            source_spec_data = np.flipud(source_spec_data)

            source_dset_name = 'source_main'

            # Arrange from slow to fast
            pos_attrs = {'units': ['nm', 'um'], 'labels': ['Y', 'X']}

            #def build_ind_val_dsets(name, inds, attrs, is_spec):

            h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices', data=source_pos_data, dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_inds, pos_attrs)

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_main_data = main_nd.reshape(np.prod(main_nd.shape[:2]),
                                               np.prod(main_nd.shape[2:]))
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
            data_utils.write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})

            # Remember to set from slow to faset
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
                                                                 lazy=False, verbose=False)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['Y', 'X', 'Cycle', 'Bias'])]))
            self.assertTrue(np.allclose(main_nd, n_dim))

            expected_n_dim = main_nd  # np.transpose(main_nd, [1, 0, 3, 2])
            n_dim, success, labels = hdf_utils.reshape_to_n_dims(
                h5_source_main, get_labels=True, sort_dims=False,
                lazy=False, verbose=False)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['Y', 'X', 'Cycle', 'Bias'])]))
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

        os.remove(file_path)

    def test_h5_beps_field(self):
        # Flipping both the spec and pos dimensions means that the order in which
        # the data is stored is the same order in which dimensions are arranged
        # In other words, sort should make no difference at all!
        file_path = 'reshape_to_n_dim_sort_required.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_raw_grp = h5_f.create_group('Raw_Measurement')

            num_rows = 3
            num_cols = 5
            num_fields = 2
            num_cycle_pts = 7
            # arrange as fast, slow
            source_pos_data = np.vstack(
                (np.tile(np.arange(num_cols), num_rows),
                 np.repeat(np.arange(num_rows), num_cols))).T
            # arrange as fast, slow
            source_spec_data = np.vstack(
                (np.tile(np.arange(num_fields), num_cycle_pts),
                 np.repeat(np.arange(num_cycle_pts), num_fields),))

            # Data is arranged from slowest to fastest

            test = np.vstack((np.arange(num_cycle_pts) * -1 - 1,
                              np.arange(num_cycle_pts) + 1))

            main_nd = np.zeros(
                shape=(num_rows, num_cols, num_fields, num_cycle_pts),
                dtype=np.float16)
            for row_ind in range(num_rows):
                for col_ind in range(num_cols):
                    main_nd[
                        row_ind, col_ind] = 1E+3 * row_ind + 1E+2 * col_ind + test

            main_nd = main_nd.transpose(0, 1, 3, 2)

            source_dset_name = 'source_main'

            # Arrange from fast to slow
            pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

            h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices',
                                                    data=source_pos_data,
                                                    dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'],
                                         is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_inds, pos_attrs)

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values',
                                                    data=source_pos_data,
                                                    dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'],
                                         is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_main_data = main_nd.reshape(np.prod(main_nd.shape[:2]),
                                               np.prod(main_nd.shape[2:]))
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name,
                                                       data=source_main_data)
            data_utils.write_safe_attrs(h5_source_main,
                                        {'units': 'A', 'quantity': 'Current'})

            # Remember to set from fast to slow
            source_spec_attrs = {'units': ['', 'V'],
                                 'labels': ['Field', 'Bias']}

            h5_source_spec_inds = h5_raw_grp.create_dataset(
                'Spectroscopic_Indices', data=source_spec_data,
                dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_source_spec_inds,
                                         source_spec_attrs['labels'],
                                         is_spec=True)
            data_utils.write_string_list_as_attr(h5_source_spec_inds,
                                                 source_spec_attrs)

            h5_source_spec_vals = h5_raw_grp.create_dataset(
                'Spectroscopic_Values', data=source_spec_data,
                dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_source_spec_vals,
                                         source_spec_attrs['labels'],
                                         is_spec=True)
            data_utils.write_string_list_as_attr(h5_source_spec_vals,
                                                 source_spec_attrs)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds,
                         h5_source_spec_vals]:
                h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(
                h5_source_main, get_labels=True, sort_dims=True,
                lazy=False, verbose=False)
            self.assertTrue(np.all(
                [x == y for x, y in zip(labels, ['Y', 'X', 'Bias', 'Field'])]))
            self.assertTrue(np.allclose(main_nd, n_dim))

            expected_n_dim = np.transpose(main_nd, [1, 0, 3, 2])
            n_dim, success, labels = hdf_utils.reshape_to_n_dims(
                h5_source_main, get_labels=True, sort_dims=False,
                lazy=False, verbose=False)
            self.assertTrue(np.all(
                [x == y for x, y in zip(labels, ['X', 'Y', 'Field', 'Bias'])]))
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

    def base_write(self, lazy_main=False, empty_main=False, pre_pos=False,
                   pre_spec=False, to_new_file=False):
        file_path = 'test.h5'
        new_file_path = 'new.h5'
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

        input_data = main_data
        kwargs = {}
        if lazy_main:
            input_data = da.from_array(main_data, chunks=main_data.shape)
        if empty_main:
            input_data = main_data.shape
            kwargs.update({'dtype': np.float16})

        with h5py.File(file_path, mode='w') as h5_f:
            if pre_spec:
                h5_spec_inds, h5_spec_vals = hdf_utils.write_ind_val_dsets(
                    h5_f, spec_dims, is_spectral=True)
                spec_dims = None
                kwargs.update({'h5_spec_inds': h5_spec_inds,
                               'h5_spec_vals': h5_spec_vals})

            if pre_pos:
                h5_pos_inds, h5_pos_vals = hdf_utils.write_ind_val_dsets(h5_f,
                                                                         pos_dims,
                                                                         is_spectral=False)
                pos_dims = None
                kwargs.update({'h5_pos_inds': h5_pos_inds,
                               'h5_pos_vals': h5_pos_vals})

            targ_loc = h5_f
            if to_new_file:
                h5_f_2 = h5py.File(new_file_path, mode='w')
                targ_loc = h5_f_2

            usid_main = hdf_utils.write_main_dataset(targ_loc, input_data, main_data_name, quantity, dset_units, pos_dims,
                                                     spec_dims, main_dset_attrs=None, slow_to_fast=False, verbose=True, **kwargs)
            self.assertIsInstance(usid_main, USIDataset)
            self.assertEqual(usid_main.name.split('/')[-1], main_data_name)
            self.assertEqual(usid_main.parent, targ_loc)
            if not empty_main:
                self.assertTrue(np.allclose(main_data, usid_main[()]))

            data_utils.validate_aux_dset_pair(self, targ_loc, usid_main.h5_pos_inds, usid_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=usid_main, is_spectral=False, slow_to_fast=False)

            data_utils.validate_aux_dset_pair(self, targ_loc, usid_main.h5_spec_inds, usid_main.h5_spec_vals, spec_names, spec_units,
                                          spec_data, h5_main=usid_main, is_spectral=True, slow_to_fast=False)

        if to_new_file:
            os.remove(new_file_path)
        os.remove(file_path)

    def test_numpy_small(self):
        self.base_write()

    def test_dask_small(self):
        self.base_write(lazy_main=True)

    def test_empty_main(self):
        self.base_write(empty_main=True)

    def test_write_main_existing_pos_aux(self):
        self.base_write(pre_pos=True, pre_spec=False)

    def test_write_main_existing_pos_aux_diff_file(self):
        self.base_write(pre_pos=True, pre_spec=False, to_new_file=True)

    def test_write_main_existing_spec_aux(self):
        self.base_write(pre_pos=False, pre_spec=True)

    def test_write_main_existing_spec_aux_diff_file(self):
        self.base_write(pre_pos=False, pre_spec=True, to_new_file=True)

    def test_write_main_both_existing_aux(self):
        self.base_write(pre_pos=True, pre_spec=True)

    def test_write_main_both_existing_aux_diff_file(self):
        self.base_write(pre_pos=True, pre_spec=True, to_new_file=True)

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

        with h5py.File(file_path, mode='w') as h5_f:
            with self.assertRaises(ValueError):
                _ = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                 spec_dims)
        os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
