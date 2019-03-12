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
from pyUSID.io import hdf_utils
from . import data_utils


if sys.version_info.major == 3:
    unicode = str


class TestHDFUtilsReg(unittest.TestCase):

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

    def test_get_indices_for_region_ref_corners(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            reg_ref = hdf_utils.get_attr(h5_main, 'even_rows')
            ret_val = hdf_utils.get_indices_for_region_ref(h5_main, reg_ref, 'corners')
            expected_pos = np.repeat(np.arange(h5_main.shape[0])[::2], 2)
            expected_spec = np.tile(np.array([0, h5_main.shape[1] - 1]), expected_pos.size // 2)
            expected_corners = np.vstack((expected_pos, expected_spec)).T
            self.assertTrue(np.allclose(ret_val, expected_corners))

    def test_get_indices_for_region_ref_slices(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            reg_ref = hdf_utils.get_attr(h5_main, 'even_rows')
            ret_val = hdf_utils.get_indices_for_region_ref(h5_main, reg_ref, 'slices')
            spec_slice = slice(0, h5_main.shape[1] - 1, None)
            expected_slices = np.array([[slice(x, x, None), spec_slice] for x in np.arange(h5_main.shape[0])[::2]])
            self.assertTrue(np.all(ret_val == expected_slices))

    def test_copy_reg_ref_reduced_dim(self):
        # TODO: Fill this test in at earliest convenience. Overriden temporarily
        assert True

    def test_clean_reg_refs_1d(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7))
            reg_ref = (slice(0, None, 2))
            cleaned = hdf_utils.clean_reg_ref(h5_dset, reg_ref)
            self.assertEqual(reg_ref, cleaned[0])
        os.remove(file_path)

    def test_clean_reg_refs_2d(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            reg_ref = (slice(0, None, 2), slice(None))
            cleaned = hdf_utils.clean_reg_ref(h5_dset, reg_ref)
            self.assertTrue(np.all([x == y for x, y in zip(reg_ref, cleaned)]))
        os.remove(file_path)

    def test_clean_reg_refs_illegal_too_many_slices(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            reg_ref = (slice(0, None, 2), slice(None), slice(1, None, 2))
            with self.assertRaises(ValueError):
                _ = hdf_utils.clean_reg_ref(h5_dset, reg_ref)

        os.remove(file_path)

    def test_clean_reg_refs_illegal_too_few_slices(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            reg_ref = (slice(0, None, 2))
            with self.assertRaises(ValueError):
                _ = hdf_utils.clean_reg_ref(h5_dset, reg_ref)

        os.remove(file_path)

    def test_clean_reg_refs_out_of_bounds(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            reg_ref = (slice(0, 13, 2), slice(None))
            expected = (slice(0, 7, 2), slice(None))
            cleaned = hdf_utils.clean_reg_ref(h5_dset, reg_ref, verbose=False)
            self.assertTrue(np.all([x == y for x, y in zip(expected, cleaned)]))
        os.remove(file_path)

    def test_attempt_reg_ref_build_spec(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(2, 5))
            dim_names = ['Bias', 'Cycle']
            expected = {'Bias': (slice(0, 1), slice(None)),
                        'Cycle': (slice(1, 2), slice(None))}
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    cleaned = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            else:
                cleaned = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            for key, value in expected.items():
                self.assertEqual(value, cleaned[key])
        os.remove(file_path)

    def test_attempt_reg_ref_build_pos(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias', 'Cycle']
            expected = {'Bias': (slice(None), slice(0, 1)),
                        'Cycle': (slice(None), slice(1, 2))}
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    cleaned = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            else:
                cleaned = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            for key, value in expected.items():
                self.assertEqual(value, cleaned[key])
        os.remove(file_path)

    def test_attempt_reg_ref_build_pos_too_many_dims(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias', 'Cycle', 'Blah']
            ret_val = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            self.assertEqual(ret_val, dict())
        os.remove(file_path)

    def test_attempt_reg_ref_build_pos_too_few_dims(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias']
            ret_val = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            self.assertEqual(ret_val, dict())
        os.remove(file_path)

    def test_write_reg_ref_main_one_dim(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(7)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2)),
                        'odd_rows': (slice(1, None, 2))}
            hdf_utils.write_region_references(h5_dset, reg_refs, add_labels_attr=True)
            self.assertEqual(len(h5_dset.attrs), 1 + len(reg_refs))
            actual = hdf_utils.get_attr(h5_dset, 'labels')
            self.assertTrue(np.all([x == y for x, y in zip(actual, ['even_rows', 'odd_rows'])]))

            expected_data = [data[0:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_reg_ref_main_1st_dim(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}
            hdf_utils.write_region_references(h5_dset, reg_refs, add_labels_attr=True)
            self.assertEqual(len(h5_dset.attrs), 1 + len(reg_refs))
            actual = hdf_utils.get_attr(h5_dset, 'labels')
            self.assertTrue(np.all([x == y for x, y in zip(actual, ['even_rows', 'odd_rows'])]))

            expected_data = [data[0:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_reg_ref_main_2nd_dim(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=data)
            reg_refs = {'even_rows': (slice(None), slice(0, None, 2)),
                        'odd_rows': (slice(None), slice(1, None, 2))}
            hdf_utils.write_region_references(h5_dset, reg_refs, add_labels_attr=False)
            self.assertEqual(len(h5_dset.attrs), len(reg_refs))
            self.assertTrue('labels' not in h5_dset.attrs.keys())

            expected_data = [data[:, 0:None:2], data[:, 1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_simple_region_ref_copy(self):
        # based on test_hdf_writer.test_write_legal_reg_ref_multi_dim_data()
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            data = np.random.rand(5, 7)
            h5_orig_dset = h5_f.create_dataset('test', data=data)
            self.assertIsInstance(h5_orig_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                'odd_rows': (slice(1, None, 2), slice(None))}}

            data_utils.write_main_reg_refs(h5_orig_dset, attrs['labels'])
            h5_f.flush()

            # two atts point to region references. one for labels
            self.assertEqual(len(h5_orig_dset.attrs), 1 + len(attrs['labels']))

            # check if the labels attribute was written:

            self.assertTrue(np.all([x in list(attrs['labels'].keys()) for x in hdf_utils.get_attr(h5_orig_dset,
                                                                                                  'labels')]))

            expected_data = [data[:None:2], data[1:None:2]]
            written_data = [h5_orig_dset[h5_orig_dset.attrs['even_rows']], h5_orig_dset[h5_orig_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

            # Now write a new dataset without the region reference:
            h5_new_dset = h5_f.create_dataset('other', data=data)
            self.assertIsInstance(h5_orig_dset, h5py.Dataset)
            h5_f.flush()

            for key in attrs['labels'].keys():
                hdf_utils.simple_region_ref_copy(h5_orig_dset, h5_new_dset, key)

            # now check to make sure that this dataset also has the same region references:
            written_data = [h5_new_dset[h5_new_dset.attrs['even_rows']], h5_new_dset[h5_new_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_create_region_ref(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Source', data=data)
            pos_inds = np.arange(0, h5_dset.shape[0], 2)
            ref_inds = [((pos_start, 0), (pos_start, h5_dset.shape[1]-1)) for pos_start in pos_inds]
            ref_inds = np.array(ref_inds)
            reg_ref = hdf_utils.create_region_reference(h5_dset, ref_inds)
            ref_slices = list()
            for start, stop in ref_inds:
                ref_slices.append([slice(start[0], stop[0]+1), slice(start[1], None)])

            h5_reg = h5_dset[reg_ref]

            h5_slice = np.vstack([h5_dset[pos_slice, spec_slice] for (pos_slice, spec_slice) in ref_slices])

            self.assertTrue(np.allclose(h5_reg, h5_slice))

        os.remove(file_path)

    def test_copy_region_refs(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(11, 7)
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_dest = h5_f.create_dataset('Target', data=data)
            source_ref = h5_dset_source.regionref[0:-1:2]
            h5_dset_source.attrs['regref'] = source_ref

            hdf_utils.copy_region_refs(h5_dset_source, h5_dset_dest)

            self.assertTrue(np.allclose(h5_dset_source[h5_dset_source.attrs['regref']],
                                        h5_dset_dest[h5_dset_dest.attrs['regref']]))

        os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
