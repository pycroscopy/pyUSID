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
from . import data_utils
sys.path.append("../../pyUSID/")
from pyUSID.io import reg_ref


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
        data_utils.make_beps_file()

    def tearDown(self):
        data_utils.delete_existing_file(data_utils.std_beps_path)

    def test_get_region_illegal_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(KeyError):
                reg_ref.get_region(h5_f['/Raw_Measurement/source_main'], 'non_existent')

    def test_get_region_legal_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_source = h5_f['/Raw_Measurement/source_main']
            returned = reg_ref.get_region(h5_source, 'even_rows')
            self.assertTrue(np.all(returned == h5_source[range(0, h5_source.shape[0], 2)]))
            
    def test_clean_reg_refs_1d(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7))
            ref_in = (slice(0, None, 2))
            cleaned = reg_ref.clean_reg_ref(h5_dset, ref_in)
            self.assertEqual(ref_in, cleaned[0])
        os.remove(file_path)

    def test_clean_reg_refs_2d(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            ref_in = (slice(0, None, 2), slice(None))
            cleaned = reg_ref.clean_reg_ref(h5_dset, ref_in)
            self.assertTrue(np.all([x == y for x, y in zip(ref_in, cleaned)]))
        os.remove(file_path)

    def test_clean_reg_refs_illegal_too_many_slices(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            ref_in = (slice(0, None, 2), slice(None), slice(1, None, 2))
            with self.assertRaises(ValueError):
                _ = reg_ref.clean_reg_ref(h5_dset, ref_in)

        os.remove(file_path)

    def test_clean_reg_refs_illegal_too_few_slices(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            ref_in = (slice(0, None, 2))
            with self.assertRaises(ValueError):
                _ = reg_ref.clean_reg_ref(h5_dset, ref_in)

        os.remove(file_path)

    def test_clean_reg_refs_out_of_bounds(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            ref_in = (slice(0, 13, 2), slice(None))
            expected = (slice(0, 7, 2), slice(None))
            cleaned = reg_ref.clean_reg_ref(h5_dset, ref_in, verbose=False)
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
                    cleaned = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            else:
                cleaned = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
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
                    cleaned = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            else:
                cleaned = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            for key, value in expected.items():
                self.assertEqual(value, cleaned[key])
        os.remove(file_path)

    def test_attempt_reg_ref_build_pos_too_many_dims(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias', 'Cycle', 'Blah']
            ret_val = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            self.assertEqual(ret_val, dict())
        os.remove(file_path)

    def test_attempt_reg_ref_build_pos_too_few_dims(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias']
            ret_val = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            self.assertEqual(ret_val, dict())
        os.remove(file_path)