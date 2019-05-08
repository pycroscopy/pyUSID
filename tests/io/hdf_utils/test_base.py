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

from tests.io import data_utils


if sys.version_info.major == 3:
    unicode = str


class TestHDFUtilsBase(unittest.TestCase):

    def setUp(self):
        data_utils.make_beps_file()

    def tearDown(self):
        data_utils.delete_existing_file(data_utils.std_beps_path)


class TestGetAttr(TestHDFUtilsBase):

    def test_not_hdf_dset(self):
        with self.assertRaises(TypeError):
            hdf_utils.get_attr(np.arange(3), 'units')

    def test_illegal_attr_type(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 14)

    def test_illegal_multiple_attrs(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], ['quantity', 'units'])

    def test_illegal_non_existent_attr(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(KeyError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 'non_existent')

    def test_legal_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            returned = hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 'units')
            self.assertEqual(returned, 'A')

    def test_legal_02(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            returned = hdf_utils.get_attr(h5_f['/Raw_Measurement/Position_Indices'], 'labels')
            self.assertTrue(np.all(returned == ['X', 'Y']))

    def test_legal_03(self):
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            for key, expected_value in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_group, key) == expected_value))


class TestGetAttributes(TestHDFUtilsBase):

    def test_one(self):
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        sub_attrs = ['att_3']
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group, sub_attrs[-1])
            self.assertIsInstance(returned_attrs, dict)
            for key in sub_attrs:
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_few(self):
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        sub_attrs = ['att_1', 'att_4', 'att_3']
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group, sub_attrs)
            self.assertIsInstance(returned_attrs, dict)
            for key in sub_attrs:
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_all(self):
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group)
            self.assertIsInstance(returned_attrs, dict)
            for key in attrs.keys():
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_absent_attr(self):
        sub_attrs = ['att_1', 'att_4', 'does_not_exist']
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_attributes(h5_group, attr_names=sub_attrs)

    def test_not_hdf_obj(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.get_attributes(np.arange(4))

    def test_invalid_type_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            with self.assertRaises(TypeError):
                _ = hdf_utils.get_attributes(h5_group, 15)

    def test_invalid_type_multi(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            with self.assertRaises(TypeError):
                _ = hdf_utils.get_attributes(h5_group, ['att_1', 15])


class TestGetAuxillaryDatasets(TestHDFUtilsBase):

    def test_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            [ret_val] = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name='Position_Indices')
            self.assertEqual(ret_val, h5_pos)

    def test_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            [ret_val] = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name='Position_Indices')
            self.assertEqual(ret_val, h5_pos)

    def test_multiple(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_pos_vals = h5_f['/Raw_Measurement/Position_Values']
            ret_val = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name=['Position_Indices',
                                                                               'Position_Values'])
            self.assertEqual(set(ret_val), set([h5_pos_inds, h5_pos_vals]))

    def test_all(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = [h5_f['/Raw_Measurement/Position_Indices'],
                        h5_f['/Raw_Measurement/Position_Values'],
                        h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Spectroscopic_Values']]
            ret_val = hdf_utils.get_auxiliary_datasets(h5_main)
            self.assertEqual(set(expected), set(ret_val))

    def test_illegal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name='Does_Not_Exist')

    def test_illegal_dset_type(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.get_auxiliary_datasets(np.arange(5), aux_dset_name='Does_Not_Exist')

    def test_illegal_target_type(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(TypeError):
                _ = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name=14)

    def test_illegal_target_type_list(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(TypeError):
                _ = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name=[14, 'Position_Indices'])


class TestGetH5ObjRefs(TestHDFUtilsBase):

    def test_many(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f,
                              4.123,
                              np.arange(6),
                              h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/Spectroscopic_Values']]
            chosen_objs = [h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices']]

            target_ref_names = ['Position_Indices', 'source_main-Fitter_000', 'Spectroscopic_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

            self.assertEqual(set(chosen_objs), set(returned_h5_objs))

    def test_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f,
                              4.123,
                              np.arange(6),
                              h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/Spectroscopic_Values']]
            chosen_objs = [h5_f['/Raw_Measurement/Position_Indices']]

            target_ref_names = ['Position_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names[0], h5_obj_refs)

            self.assertEqual(set(chosen_objs), set(returned_h5_objs))

    def test_non_string_names(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f, 4.123, np.arange(6),
                           h5_f['/Raw_Measurement/Position_Indices']]

            target_ref_names = ['Position_Indices', np.arange(6), 4.123]

            with self.assertRaises(TypeError):
                _ = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

    def test_no_hdf5_datasets(self):
        h5_obj_refs = 4.124

        target_ref_names = ['Position_Indices']

        with self.assertRaises(TypeError):
            _ = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

    def test_same_name(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f['/Raw_Measurement/source_main-Fitter_001/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/Spectroscopic_Values']]
            expected_objs = [h5_f['/Raw_Measurement/source_main-Fitter_001/Spectroscopic_Indices'],
                             h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices']]

            target_ref_names = ['Spectroscopic_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

            self.assertEqual(set(expected_objs), set(returned_h5_objs))


class TestFindDataset(TestHDFUtilsBase):

    def test_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/']
            expected_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/Spectroscopic_Indices']]
            ret_val = hdf_utils.find_dataset(h5_group, 'Spectroscopic_Indices')
            self.assertEqual(set(ret_val), set(expected_dsets))


class TestWriteSimpleAttrs(TestHDFUtilsBase):

    def test_invalid_h5_obj(self):
        with self.assertRaises(TypeError):
            hdf_utils.write_simple_attrs(np.arange(4), {'sds': 3})

    def test_invalid_h5_obj_reg_ref(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            ref_in = hdf_utils.get_attr(h5_main, 'even_rows')
            with self.assertRaises(TypeError):
                hdf_utils.write_simple_attrs(ref_in, {'sds': 3})

    def test_invalid_attrs_dict(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_group = h5_f.create_group('Blah')
            with self.assertRaises(TypeError):
                hdf_utils.write_simple_attrs(h5_group, ['attrs', 1.234, 'should be dict', np.arange(3)])

    def test_to_grp(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            h5_group = h5_f.create_group('Blah')

            attrs = {'att_1': 'string_val', 'att_2': 1.234, 'att_3': [1, 2, 3.14, 4],
                     'att_4': ['s', 'tr', 'str_3']}

            hdf_utils.write_simple_attrs(h5_group, attrs)

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_group, key) == expected_val))

        os.remove(file_path)

    def test_np_array(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            attrs = {'att_1': np.random.rand(4)}

            hdf_utils.write_simple_attrs(h5_f, attrs)

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_f, key) == expected_val))

        os.remove(file_path)

    def test_none_ignored(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            attrs = {'att_1': None}

            hdf_utils.write_simple_attrs(h5_f, attrs)

            self.assertTrue('att_1' not in h5_f.attrs.keys())

        os.remove(file_path)

    def test_to_dset(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            h5_dset = h5_f.create_dataset('Test', data=np.arange(3))

            attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3']}

            hdf_utils.write_simple_attrs(h5_dset, attrs)

            self.assertEqual(len(h5_dset.attrs), len(attrs))

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_dset, key) == expected_val))

        os.remove(file_path)


class TestIsEditableH5(TestHDFUtilsBase):

    def test_read_only(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            h5_main = h5_f['/Raw_Measurement/Ancillary']
            self.assertFalse(hdf_utils.is_editable_h5(h5_group))
            self.assertFalse(hdf_utils.is_editable_h5(h5_f))
            self.assertFalse(hdf_utils.is_editable_h5(h5_main))

    def test_r_plus(self):
        with h5py.File(data_utils.std_beps_path, mode='r+') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            h5_main = h5_f['/Raw_Measurement/Ancillary']
            self.assertTrue(hdf_utils.is_editable_h5(h5_group))
            self.assertTrue(hdf_utils.is_editable_h5(h5_f))
            self.assertTrue(hdf_utils.is_editable_h5(h5_main))

    def test_w(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.arange(3))
            h5_group = h5_f.create_group('blah')
            self.assertTrue(hdf_utils.is_editable_h5(h5_group))
            self.assertTrue(hdf_utils.is_editable_h5(h5_f))
            self.assertTrue(hdf_utils.is_editable_h5(h5_dset))

        os.remove(file_path)

    def test_invalid_type(self):
        # wrong kind of object
        with self.assertRaises(TypeError):
            _ = hdf_utils.is_editable_h5(np.arange(4))

    def test_closed_file(self):
        with h5py.File(data_utils.std_beps_path, mode='r+') as h5_f:
            h5_group = h5_f['/Raw_Measurement']

        with self.assertRaises(ValueError):
            _ = hdf_utils.is_editable_h5(h5_group)


class TestLinkH5ObjAsAlias(TestHDFUtilsBase):

    def test_legal(self):
        file_path = 'link_as_alias.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            h5_main = h5_f.create_dataset('main', data=np.arange(5))
            h5_anc = h5_f.create_dataset('Ancillary', data=np.arange(3))
            h5_group = h5_f.create_group('Results')

            # Linking to dataset:
            hdf_utils.link_h5_obj_as_alias(h5_main, h5_anc, 'Blah')
            hdf_utils.link_h5_obj_as_alias(h5_main, h5_group, 'Something')
            self.assertEqual(h5_f[h5_main.attrs['Blah']], h5_anc)
            self.assertEqual(h5_f[h5_main.attrs['Something']], h5_group)

            # Linking ot Group:
            hdf_utils.link_h5_obj_as_alias(h5_group, h5_main, 'Center')
            hdf_utils.link_h5_obj_as_alias(h5_group, h5_anc, 'South')
            self.assertEqual(h5_f[h5_group.attrs['Center']], h5_main)
            self.assertEqual(h5_f[h5_group.attrs['South']], h5_anc)

            # Linking to file:
            hdf_utils.link_h5_obj_as_alias(h5_f, h5_main, 'Paris')
            hdf_utils.link_h5_obj_as_alias(h5_f, h5_group, 'France')
            self.assertEqual(h5_f[h5_f.attrs['Paris']], h5_main)
            self.assertEqual(h5_f[h5_f.attrs['France']], h5_group)

    def test_not_h5_obj(self):
        file_path = 'link_as_alias.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_group = h5_f.create_group('Results')

            # Non h5 object
            with self.assertRaises(TypeError):
                hdf_utils.link_h5_obj_as_alias(h5_group, np.arange(5), 'Center')

            # H5 reference but not the object
            with self.assertRaises(TypeError):
                hdf_utils.link_h5_obj_as_alias('not_a_dset', h5_group, 'Center')

            with self.assertRaises(TypeError):
                hdf_utils.link_h5_obj_as_alias(h5_group, h5_group, 1.234)

        os.remove(file_path)


class TestLinkH5ObjectAsAttribute(TestHDFUtilsBase):

    def test_legal(self):
        file_path = 'link_h5_objects_as_attrs.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            h5_main = h5_f.create_dataset('main', data=np.arange(5))
            h5_anc = h5_f.create_dataset('Ancillary', data=np.arange(3))
            h5_group = h5_f.create_group('Results')

            hdf_utils.link_h5_objects_as_attrs(h5_f, [h5_anc, h5_main, h5_group])
            for exp, name in zip([h5_main, h5_anc, h5_group], ['main', 'Ancillary', 'Results']):
                self.assertEqual(exp, h5_f[h5_f.attrs[name]])

            # Single object
            hdf_utils.link_h5_objects_as_attrs(h5_main, h5_anc)
            self.assertEqual(h5_f[h5_main.attrs['Ancillary']], h5_anc)

            # Linking to a group:
            hdf_utils.link_h5_objects_as_attrs(h5_group, [h5_anc, h5_main])
            for exp, name in zip([h5_main, h5_anc], ['main', 'Ancillary']):
                self.assertEqual(exp, h5_group[h5_group.attrs[name]])

        os.remove(file_path)

    def test_wrong_type(self):
        file_path = 'link_h5_objects_as_attrs.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_main = h5_f.create_dataset('main', data=np.arange(5))

            with self.assertRaises(TypeError):
                hdf_utils.link_h5_objects_as_attrs(h5_main, np.arange(4))

            with self.assertRaises(TypeError):
                hdf_utils.link_h5_objects_as_attrs(np.arange(4), h5_main)

        os.remove(file_path)


class TestWriteBookKeepingAttrs(TestHDFUtilsBase):

    def test_file(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            hdf_utils.write_book_keeping_attrs(h5_f)
            data_utils.verify_book_keeping_attrs (self, h5_f)
        os.remove(file_path)

    def test_group(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_g = h5_f.create_group('group')
            hdf_utils.write_book_keeping_attrs(h5_g)
            data_utils.verify_book_keeping_attrs (self, h5_g)
        os.remove(file_path)

    def test_dset(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('dset', data=[1, 2, 3])
            hdf_utils.write_book_keeping_attrs(h5_dset)
            data_utils.verify_book_keeping_attrs (self, h5_dset)
        os.remove(file_path)

    def test_invalid(self):
        with self.assertRaises(TypeError):
            hdf_utils.write_book_keeping_attrs(np.arange(4))


if __name__ == '__main__':
    unittest.main()
