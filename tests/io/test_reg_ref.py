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