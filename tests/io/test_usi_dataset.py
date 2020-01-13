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
import matplotlib as mpl
# Attempting to get things to work for all versions of python on Travis
mpl.use('Agg')
sys.path.append("../../pyUSID/")
from pyUSID.io import USIDataset, hdf_utils
from pyUSID.io.write_utils import Dimension

from . import data_utils

skip_viz_tests = True
if sys.version_info.major == 3:
    unicode = str
    if sys.version_info.minor > 4:
        skip_viz_tests = False

test_h5_file_path = data_utils.std_beps_path


class TestBEPS(unittest.TestCase):

    def setUp(self):
        data_utils.make_beps_file()
        self.orig_labels_order = ['X', 'Y', 'Cycle', 'Bias']
        self.h5_file = h5py.File(data_utils.std_beps_path, mode='r')

        h5_grp = self.h5_file['/Raw_Measurement/']
        self.source_nd_s2f = h5_grp['n_dim_form'][()]
        self.source_nd_f2s = self.source_nd_s2f.transpose(1, 0, 3, 2)
        self.h5_source = USIDataset(h5_grp['source_main'])

        self.pos_dims=[]
        self.spec_dims=[]

        for dim_name, dim_units in zip(self.h5_source.pos_dim_labels,
                                       hdf_utils.get_attr(self.h5_source.h5_pos_inds, 'units')):
            self.pos_dims.append(
                Dimension(dim_name, dim_units, h5_grp[dim_name][()]))

        for dim_name, dim_units in zip(self.h5_source.spec_dim_labels,
                                       hdf_utils.get_attr(self.h5_source.h5_spec_inds, 'units')):
            self.spec_dims.append(
                Dimension(dim_name, dim_units, h5_grp[dim_name][()]))

        res_grp_0 = h5_grp['source_main-Fitter_000']
        self.results_0_nd_s2f = res_grp_0['n_dim_form'][()]
        self.results_0_nd_f2s = self.results_0_nd_s2f.transpose(1, 0, 3, 2)
        self.h5_compound = USIDataset(res_grp_0['results_main'])

        res_grp_1 = h5_grp['source_main-Fitter_001']
        self.results_1_nd_s2f = res_grp_1['n_dim_form'][()]
        self.results_1_nd_f2s = self.results_1_nd_s2f.transpose(1, 0, 3, 2)
        self.h5_complex = USIDataset(res_grp_1['results_main'])

    def tearDown(self):
        self.h5_file.close()
        os.remove(data_utils.std_beps_path)


class TestUSIDatasetReal(unittest.TestCase):

    def setUp(self):
        self.rev_spec = False
        data_utils.make_beps_file(rev_spec=self.rev_spec)
        self.orig_labels_order = ['X', 'Y', 'Cycle', 'Bias'] if self.rev_spec else ['X', 'Y', 'Bias', 'Cycle']

    def tearDown(self):
        os.remove(test_h5_file_path)

    def get_expected_n_dim(self, h5_f):
        nd_slow_to_fast = h5_f['/Raw_Measurement/n_dim_form'][()]
        nd_fast_to_slow = nd_slow_to_fast.transpose(1, 0, 3, 2)
        if self.rev_spec:
            nd_fast_to_slow = nd_fast_to_slow.transpose(0, 1, 3, 2)
        return nd_slow_to_fast, nd_fast_to_slow


class TestStringRepr(TestBEPS):

    def test_string_representation(self):
        usi_dset = self.h5_source
        h5_main = self.h5_file[usi_dset.name]
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


class TestEquality(TestBEPS):

    def test_correct_USIDataset(self):
        expected = USIDataset(self.h5_source)
        self.assertTrue(expected == expected)

    def test_correct_h5_dataset(self):
        h5_main = self.h5_file[self.h5_source.name]
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


class TestGetNDimFormExistsReal(TestUSIDatasetReal):

    def test_sorted_and_unsorted(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_dset = USIDataset(h5_f['/Raw_Measurement/source_main'])
            nd_slow_to_fast, nd_fast_to_slow = self.get_expected_n_dim(h5_f)
            actual_f2s = usi_dset.get_n_dim_form(lazy=False)
            self.assertTrue(np.allclose(nd_fast_to_slow, actual_f2s))

            nd_form, success = hdf_utils.reshape_to_n_dims(usi_dset, sort_dims=True)
            print(nd_form.shape)

            usi_dset.toggle_sorting()
            actual_s2f = usi_dset.get_n_dim_form(lazy=False)
            self.assertTrue(np.allclose(nd_slow_to_fast, actual_s2f))


class TestPosSpecSlicesReal(TestUSIDatasetReal):

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
            with self.assertRaises(IndexError):
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


class TestGetUnitValuesReal(TestUSIDatasetReal):

    def test_get_pos_values(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            for dim_name in ['X', 'Y']:
                expected = h5_f['/Raw_Measurement/' + dim_name][()]
                actual = usi_main.get_pos_values(dim_name)
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
            for dim_name in ['Bias', 'Cycle']:
                expected = h5_f['/Raw_Measurement/' + dim_name][()]
                actual = usi_main.get_spec_values(dim_name)
                self.assertTrue(np.allclose(expected, actual))

    def test_get_spec_values_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = usi_main.get_spec_values('blah')
            with self.assertRaises(TypeError):
                _ = usi_main.get_spec_values(np.array(5))


class TestSliceReal(TestUSIDatasetReal):

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

    def test_out_of_bounds(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(IndexError):
                _ = usi_main.slice({'X': 15, 'Y': 1})

    def base(self, slice_dict, f2s_slice_list, result_as_nd, lazy_result,
             verbose=False):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = usi_main.slice(slice_dict,
                                             ndim_form=result_as_nd,
                                             lazy=lazy_result)
            if verbose:
                print('Status: {}, actual.shape: {}, actual.dtype: {}, '
                      'type(actual): {}'.format(success, actual.shape,
                                                actual.dtype, type(actual)))

            self.assertTrue(success)
            n_dim_s2f, n_dim_f2s = self.get_expected_n_dim(h5_f)

            if result_as_nd:
                expected = n_dim_f2s[tuple(f2s_slice_list)]
                expected = expected.squeeze()
            else:
                s2f_slice_list = f2s_slice_list[:2][::-1] + \
                                 f2s_slice_list[2:][::-1]
                if verbose:
                    print('Slice list converted from: {} to {}'
                          ''.format(f2s_slice_list, s2f_slice_list))

                expected = n_dim_s2f[tuple(s2f_slice_list)]
                if verbose:
                    print('Expected in N-dim form: {}'.format(expected.shape))

                expected = expected.reshape(np.prod(expected.shape[:2]),
                                            np.prod(expected.shape[2:]))
                if verbose:
                    print('Expected after flattening of shape: {}'
                          ''.format(expected.shape))

            if lazy_result:
                self.assertIsInstance(actual, da.core.Array)
                actual = actual.compute()

            self.assertTrue(np.allclose(expected, actual))

    def test_empty_2d_numpy(self):
        self.base(None, [slice(None) for _ in range(4)], False, False)

    def test_empty_nd_numpy(self):
        self.base(None, [slice(None) for _ in range(4)], True, False)

    def test_empty_nd_dask(self):
        self.base(None, [slice(None) for _ in range(4)], True, True)

    def test_empty_2d_dask(self):
        self.base(None, [slice(None) for _ in range(4)], False, True)

    def test_negative_index_nd_numpy(self):
        self.base({'X': -2, 'Y': 1},
                  [slice(-2, -1), slice(1, 2)] + [slice(None) for _ in range(2)],
                  True, False)

    def test_negative_index_nd_dask(self):
        self.base({'X': -2, 'Y': 1},
                  [slice(-2, -1), slice(1, 2)] + [slice(None) for _ in range(2)],
                  True, True)

    def test_negative_index_2d_numpy(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = usi_main.slice({'X': -2, 'Y': 1}, ndim_form=False)

    def test_one_pos_dim_removed_nd_numpy(self):
        self.base({'X': 3},
                  [3] + [slice(None) for _ in range(3)], True, False)

    def test_one_pos_dim_removed_nd_dask(self):
        self.base({'X': 3},
                  [3] + [slice(None) for _ in range(3)], True, True)

    def test_one_pos_dim_removed_2d_numpy(self):
        self.base({'X': 3},
                  [slice(3, 4)] + [slice(None) for _ in range(3)],
                  False, False)

    def test_one_pos_dim_removed_2d_dask(self):
        self.base({'X': 3},
                  [slice(3, 4)] + [slice(None) for _ in range(3)],
                  False, True)

    def test_one_pos_dim_sliced_nd_numpy(self):
        self.base({'X': slice(1, 5, 2)},
                  [slice(1, 5, 2)] + [slice(None) for _ in range(3)],
                  True, False)

    def test_one_pos_dim_sliced_nd_dask(self):
        self.base({'X': slice(1, 5, 2)},
                  [slice(1, 5, 2)] + [slice(None) for _ in range(3)],
                  True, True)

    def test_one_pos_dim_sliced_2d_numpy(self):
        self.base({'X': slice(1, 5, 2)},
                  [slice(1, 5, 2)] + [slice(None) for _ in range(3)],
                  False, False)

    def test_one_pos_dim_sliced_2d_dask(self):
        self.base({'X': slice(1, 5, 2)},
                  [slice(1, 5, 2)] + [slice(None) for _ in range(3)],
                  False, True)

    def test_two_pos_dim_sliced_nd_numpy(self):
        self.base({'X': slice(1, 5, 2), 'Y': 1},
                  [slice(1, 5, 2), slice(1, 2)] + [slice(None) for _ in range(2)],
                  True, False)

    def test_two_pos_dim_sliced_nd_dask(self):
        self.base({'X': slice(1, 5, 2), 'Y': 1},
                  [slice(1, 5, 2), slice(1, 2)] + [slice(None) for _ in range(2)],
                  True, True)

    def test_two_pos_dim_sliced_2d_numpy(self):
        self.base({'X': slice(1, 5, 2), 'Y': 1},
                  [slice(1, 5, 2), slice(1, 2)] + [slice(None) for _ in range(2)],
                  False, False)

    def test_two_pos_dim_sliced_2d_dask(self):
        self.base({'X': slice(1, 5, 2), 'Y': 1},
                  [slice(1, 5, 2), slice(1, 2)] + [slice(None) for _ in range(2)],
                  False, True)

    def test_two_pos_dim_sliced_list_nd_numpy(self):
        self.base({'X': [1, 2, 4], 'Y': 1},
                  [[1, 2, 4], slice(1, 2)] + [slice(None) for _ in range(2)],
                  True, False)

    def test_two_pos_dim_sliced_list_nd_dask(self):
        self.base({'X': [1, 2, 4], 'Y': 1},
                  [[1, 2, 4], slice(1, 2)] + [slice(None) for _ in range(2)],
                  True, True)

    def test_two_pos_dim_sliced_list_2d_numpy(self):
        self.base({'X': [1, 2, 4], 'Y': 1},
                  [[1, 2, 4], slice(1, 2)] + [slice(None) for _ in range(2)],
                  False, False)

    def test_two_pos_dim_sliced_list_2d_dask(self):
        self.base({'X': [1, 2, 4], 'Y': 1},
                  [[1, 2, 4], slice(1, 2)] + [slice(None) for _ in range(2)],
                  False, True)

    def test_both_pos_removed_nd_numpy(self):
        self.base({'X': 3, 'Y': 1},
                  [slice(3, 4), slice(1, 2)] + [slice(None) for _ in range(2)],
                  True, False)

    def test_both_pos_removed_nd_dask(self):
        self.base({'X': 3, 'Y': 1},
                  [slice(3, 4), slice(1, 2)] + [slice(None) for _ in range(2)],
                  True, True)

    def test_both_pos_removed_2d_numpy(self):
        self.base({'X': 3, 'Y': 1},
                  [slice(3, 4), slice(1, 2)] + [slice(None) for _ in range(2)],
                  False, False)

    def test_both_pos_removed_2d_dask(self):
        self.base({'X': 3, 'Y': 1},
                  [slice(3, 4), slice(1, 2)] + [slice(None) for _ in range(2)],
                  False, True)

    def test_pos_and_spec_sliced_list_nd_numpy(self):
        self.base({'X': [1, 2, 4], 'Bias': slice(1, 7, 3)},
                  [[1, 2, 4], slice(None), slice(1, 7, 3), slice(None)],
                  True, False)

    def test_pos_and_spec_sliced_list_nd_dask(self):
        self.base({'X': [1, 2, 4], 'Bias': slice(1, 7, 3)},
                  [[1, 2, 4], slice(None), slice(1, 7, 3), slice(None)],
                  True, True)

    def test_pos_and_spec_sliced_list_2d_numpy(self):
        self.base({'X': [1, 2, 4], 'Bias': slice(1, 7, 3)},
                  [[1, 2, 4], slice(None), slice(1, 7, 3), slice(None)],
                  False, False)

    def test_pos_and_spec_sliced_list_2d_dask(self):
        self.base({'X': [1, 2, 4], 'Bias': slice(1, 7, 3)},
                  [[1, 2, 4], slice(None), slice(1, 7, 3), slice(None)],
                  False, True)

    def test_all_dims_sliced_nd_numpy(self):
        self.base({'X': [1, 2, 4], 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1},
                  [[1, 2, 4], slice(2, 3), slice(1, 7, 3), slice(1, 2)],
                  True, False)

    def test_all_dims_sliced_nd_dask(self):
        self.base({'X': [1, 2, 4], 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1},
                  [[1, 2, 4], slice(2, 3), slice(1, 7, 3), slice(1, 2)],
                  True, True)

    def test_all_dims_sliced_2d_numpy(self):
        self.base({'X': [1, 2, 4], 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1},
                  [[1, 2, 4], slice(2, 3), slice(1, 7, 3), slice(1, 2)],
                  False, False)

    def test_all_dims_sliced_2d_dask(self):
        self.base({'X': [1, 2, 4], 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1},
                  [[1, 2, 4], slice(2, 3), slice(1, 7, 3), slice(1, 2)],
                  False, True)

    def test_all_but_one_dims_sliced_nd_numpy(self):
        self.base({'X': 1, 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1},
                  [slice(1, 2), slice(2, 3), slice(1, 7, 3), slice(1, 2)],
                  True, False)

    def test_all_but_one_dims_sliced_nd_dask(self):
        self.base({'X': 1, 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1},
                  [slice(1, 2), slice(2, 3), slice(1, 7, 3), slice(1, 2)],
                  True, True)

    def test_all_but_one_dims_sliced_2d_numpy(self):
        self.base({'X': 1, 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1},
                  [slice(1, 2), slice(2, 3), slice(1, 7, 3), slice(1, 2)],
                  False, False)

    def test_all_but_one_dims_sliced_2d_dask(self):
        self.base({'X': 1, 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1},
                  [slice(1, 2), slice(2, 3), slice(1, 7, 3), slice(1, 2)],
                  False, True)

    def test_all_dims_sliced_nd_numpy(self):
        self.base({'X': 1, 'Y': 2, 'Bias': 4, 'Cycle': 1},
                  [slice(1, 2), slice(2, 3), slice(4, 5), slice(1, 2)],
                  True, False)

    def test_all_dims_sliced_nd_dask(self):
        self.base({'X': 1, 'Y': 2, 'Bias': 4, 'Cycle': 1},
                  [slice(1, 2), slice(2, 3), slice(4, 5), slice(1, 2)],
                  True, True)

    def test_all_dims_sliced_2d_dask(self):
        self.base({'X': 1, 'Y': 2, 'Bias': 4, 'Cycle': 1},
                  [slice(1, 2), slice(2, 3), slice(4, 5), slice(1, 2)],
                  False, False)

    def test_all_dims_sliced_2d_dask(self):
        self.base({'X': 1, 'Y': 2, 'Bias': 4, 'Cycle': 1},
                  [slice(1, 2), slice(2, 3), slice(4, 5), slice(1, 2)],
                  False, True)


class TestSortingReal(TestUSIDatasetReal):

    def test_toggle_sorting(self):
        # Need to change data file so that sorting actually does something
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])

            self.assertEqual(usi_main.n_dim_labels, self.orig_labels_order)

            usi_main.toggle_sorting()

            self.assertEqual(usi_main.n_dim_labels, ['Y', 'X', 'Cycle', 'Bias'])

            usi_main.toggle_sorting()

            self.assertEqual(usi_main.n_dim_labels, self.orig_labels_order)

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


class TestGetDimsForSliceReal(TestUSIDatasetReal):

    @staticmethod
    def get_all_dimensions():
        pos_dims = []
        spec_dims = []
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_raw_grp = h5_f['Raw_Measurement']
            usi_main = USIDataset(h5_raw_grp['source_main'])
            for dim_name, dim_units in zip(usi_main.pos_dim_labels,
                                           hdf_utils.get_attr(usi_main.h5_pos_inds, 'units')):
                pos_dims.append(Dimension(dim_name, dim_units, h5_raw_grp[dim_name][()]))

            for dim_name, dim_units in zip(usi_main.spec_dim_labels,
                                           hdf_utils.get_attr(
                                               usi_main.h5_spec_inds, 'units')):
                spec_dims.append(Dimension(dim_name, dim_units, h5_raw_grp[dim_name][()]))

        return pos_dims, spec_dims

    def setUp(self):
        super(TestGetDimsForSliceReal, self).setUp()
        self.pos_dims, self.spec_dims = self.get_all_dimensions()
        self.default_dimension = Dimension('arb.', 'a. u.', [1])
        self.pos_dict = dict()
        self.spec_dict = dict()
        for item in self.pos_dims:
            self.pos_dict[item.name] = item
        for item in self.spec_dims:
            self.spec_dict[item.name] = item

    def __validate_dim_list(self, expected, actual):
        self.assertIsInstance(expected, list)
        self.assertIsInstance(expected, list)
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertEqual(left, right)

    def base(self, slice_dict, pos_exp, spec_exp, verbose=False):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            usi_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            pos_act, spec_act = usi_main._get_dims_for_slice(slice_dict=slice_dict,
                                                             verbose=verbose)
        if verbose:
            print(pos_act)
            print(spec_act)

        self.__validate_dim_list(pos_act, pos_exp)
        self.__validate_dim_list(spec_act, spec_exp)

    def test_empty(self):
        self.base(None, self.pos_dims, self.spec_dims)

    def test_single_pos_dim_sliced(self):
        self.base({'X': 2}, [self.pos_dict['Y']], self.spec_dims)

    def test_single_pos_dim_truncated(self):
        new_pos_dims = list()
        for item in self.pos_dims:
            if item.name == 'X':
                new_pos_dims.append(Dimension(item.name, item.units, item.values[slice(1, 5, 2)]))
            else:
                new_pos_dims.append(item)

        self.base({'X': slice(1, 5, 2)}, new_pos_dims, self.spec_dims)

    def test_both_pos_dim_sliced(self):
        self.base({'X': 2, 'Y': 0}, [self.default_dimension], self.spec_dims)

    def test_single_spec_dim_sliced(self):
        self.base({'Bias': 2}, self.pos_dims, [self.spec_dict['Cycle']])

    def test_single_spec_dim_truncated(self):
        new_spec_dims = list()
        for item in self.spec_dims:
            if item.name == 'Bias':
                new_spec_dims.append(Dimension(item.name, item.units, item.values[slice(1, 7, 3)]))
            else:
                new_spec_dims.append(item)

        self.base({'Bias': slice(1, 7, 3)}, self.pos_dims, new_spec_dims)

    def test_both_spec_dim_sliced(self):
        self.base({'Bias': 4, 'Cycle': 1}, self.pos_dims, [self.default_dimension],
                  verbose=False)

    def test_one_pos_one_spec_dims_sliced(self):
        self.base({'X': 1, 'Bias': 2}, [self.pos_dict['Y']], [self.spec_dict['Cycle']])

    def test_all_dims_sliced(self):
        self.base({'X': 1, 'Y': 2, 'Bias': 4, 'Cycle': 1},
                  [self.default_dimension], [self.default_dimension])


def get_tick_labels(tick_labels):
    return np.array([float(x.get_text()) for x in tick_labels])


def validate_imshow(self, axis, exp_data, title=None, x_vec=None, y_vec=None,
                    x_label=None, y_label=None, verbose=False):
    self.assertIsInstance(axis, mpl.axes.Axes)
    im_handles = [obj for obj in axis.get_children() if
                  isinstance(obj, mpl.image.AxesImage)]
    self.assertEqual(len(im_handles), 1)
    im_handle = im_handles[0]
    actual_data = im_handle.get_array().data
    if verbose:
        print(actual_data.shape, exp_data.shape)
    self.assertTrue(np.allclose(actual_data, exp_data))
    if title is not None:
        self.assertEqual(axis.get_title(), title)
    if x_label is not None:
        self.assertEqual(axis.get_xlabel(), x_label)
    if y_label is not None:
        self.assertEqual(axis.get_ylabel(), y_label)
    self.assertEqual(axis.get_xscale(), 'linear')

    x_ref = get_tick_labels(axis.get_xticklabels())
    y_ref = get_tick_labels(axis.get_yticklabels())
    self.assertAlmostEqual(x_ref[0], np.round(x_vec[0], 2))
    self.assertAlmostEqual(x_ref[-1], np.round(x_vec[-1], 2))
    self.assertAlmostEqual(y_ref[0], np.round(y_vec[0], 2))
    self.assertAlmostEqual(y_ref[-1], np.round(y_vec[-1], 2))


def validate_single_curve(self, axis, x_vec, y_vec, title=None, x_label=None,
                          y_label=None):
    self.assertIsInstance(axis, mpl.axes.Axes)
    line_handles = [obj for obj in axis.get_children() if
                  isinstance(obj, mpl.lines.Line2D)]
    self.assertEqual(len(line_handles), 1)
    line_handle = line_handles[0]
    # for each curve in the plot:
    self.assertTrue(np.allclose(line_handle.get_xdata(), x_vec))
    self.assertTrue(np.allclose(line_handle.get_ydata(), y_vec))
    # verify legend
    if x_label is not None:
        self.assertEqual(axis.get_xlabel(), x_label)
    if y_label is not None:
        self.assertEqual(axis.get_ylabel(), y_label)
    if title is not None:
        self.assertEqual(axis.get_title(), title)
    # verify fig suptitles

"""
def validate_subplots(axes):
    pass
"""


class TestSimpleStaticVisualizationReal(TestUSIDatasetReal):
    
    def test_two_pos_simple(self):
        if skip_viz_tests: return
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            dset_path = '/Raw_Measurement/source_main'
            usi_main = USIDataset(h5_f[dset_path])
            slice_dict = {'Bias': 0, 'Cycle': 1}
            exp_data, success = usi_main.slice(slice_dict=slice_dict)
            self.assertTrue(success)
            fig, axis = usi_main.visualize(slice_dict=slice_dict)
            validate_imshow(self, axis, exp_data, title=dset_path,
                            x_vec=h5_f['/Raw_Measurement/' + usi_main.pos_dim_labels[1]],
                            y_vec=h5_f['/Raw_Measurement/' + usi_main.pos_dim_labels[0]],
                            x_label=usi_main.pos_dim_descriptors[1],
                            y_label=usi_main.pos_dim_descriptors[0])
    
    def test_two_spec(self):
        if skip_viz_tests: return
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            dset_path = '/Raw_Measurement/source_main'
            usi_main = USIDataset(h5_f[dset_path])
            slice_dict = {'X': 3, 'Y': 2}
            exp_data, success = usi_main.slice(slice_dict=slice_dict)
            self.assertTrue(success)
            fig, axis = usi_main.visualize(slice_dict=slice_dict)
            validate_imshow(self, axis, exp_data, title=dset_path,
                            x_vec=h5_f['/Raw_Measurement/' + usi_main.spec_dim_labels[1]],
                            y_vec=h5_f['/Raw_Measurement/' + usi_main.spec_dim_labels[0]],
                            x_label=usi_main.spec_dim_descriptors[1],
                            y_label=usi_main.spec_dim_descriptors[0])

    def test_one_pos_one_spec(self):
        if skip_viz_tests: return
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            dset_path = '/Raw_Measurement/source_main'
            usi_main = USIDataset(h5_f[dset_path])
            slice_dict = {'X': 3, 'Bias': 2}
            exp_data, success = usi_main.slice(slice_dict=slice_dict)
            self.assertTrue(success)
            fig, axis = usi_main.visualize(slice_dict=slice_dict)
            spec_ind = usi_main.spec_dim_labels.index('Cycle')
            pos_ind = usi_main.pos_dim_labels.index('Y')
            validate_imshow(self, axis, exp_data, title=dset_path,
                            x_vec=h5_f['/Raw_Measurement/' + usi_main.spec_dim_labels[spec_ind]],
                            y_vec=h5_f['/Raw_Measurement/' + usi_main.pos_dim_labels[pos_ind]],
                            x_label=usi_main.spec_dim_descriptors[spec_ind],
                            y_label=usi_main.pos_dim_descriptors[pos_ind])

    def test_one_pos(self):
        if skip_viz_tests: return
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            dset_path = '/Raw_Measurement/source_main'
            usi_main = USIDataset(h5_f[dset_path])
            slice_dict = {'Bias': 4, 'Cycle': 1, 'Y': 2}
            rem_dim_name = 'X'
            pos_ind = usi_main.pos_dim_labels.index(rem_dim_name)
            exp_data, success = usi_main.slice(slice_dict=slice_dict)
            self.assertTrue(success)
            fig, axis = usi_main.visualize(slice_dict=slice_dict)
            validate_single_curve(self, axis, h5_f['/Raw_Measurement/' + rem_dim_name],
                                  exp_data,
                                  title=dset_path,
                                  x_label=usi_main.pos_dim_descriptors[pos_ind],
                                  y_label=usi_main.data_descriptor)
    
    def test_one_spec(self):
        if skip_viz_tests: return
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            dset_path = '/Raw_Measurement/source_main'
            usi_main = USIDataset(h5_f[dset_path])
            slice_dict = {'Bias': 4, 'X': 1, 'Y': 2}
            rem_dim_name = 'Cycle'
            spec_ind = usi_main.spec_dim_labels.index(rem_dim_name)
            exp_data, success = usi_main.slice(slice_dict=slice_dict)
            self.assertTrue(success)
            fig, axis = usi_main.visualize(slice_dict=slice_dict)
            validate_single_curve(self, axis, h5_f['/Raw_Measurement/' + rem_dim_name],
                                  exp_data,
                                  title=dset_path,
                                  x_label=usi_main.spec_dim_descriptors[spec_ind],
                                  y_label=usi_main.data_descriptor)

"""
    def test_no_dims(self):
        pass
    
    def test_more_than_2_dims(self):
        pass



class TestComplexStaticVisualization(unittest.TestCase):
    
    def setUp(self):
        self.file_path = 'complex_viz.h5'
        pass
        
    def tearDown(self):
        os.remove(self.file_path)
    
    def test_image(self):
        pass

    def test_curve(self):
        pass


class TestCompoundStaticVisualization(unittest.TestCase):
    
    def setUp(self):
        self.file_path = 'complex_viz.h5'
        pass
        
    def tearDown(self):
        os.remove(self.file_path)
    
    def test_image(self):
        pass

    def test_curve(self):
        pass
"""


if __name__ == '__main__':
    unittest.main()
