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

from .. import data_utils

if sys.version_info.major == 3:
    unicode = str


class TestHDFUtilsBase(unittest.TestCase):

    def setUp(self):
        data_utils.make_beps_file()

    def tearDown(self):
        data_utils.delete_existing_file(data_utils.std_beps_path)


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


class TestWriteBookKeepingAttrs(unittest.TestCase):

    def test_file(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            hdf_utils.write_book_keeping_attrs(h5_f)
            data_utils.verify_book_keeping_attrs (self, h5_f)
        os.remove(file_path)

    def test_group(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_g = h5_f.create_group('group')
            hdf_utils.write_book_keeping_attrs(h5_g)
            data_utils.verify_book_keeping_attrs (self, h5_g)
        os.remove(file_path)

    def test_dset(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('dset', data=[1, 2, 3])
            hdf_utils.write_book_keeping_attrs(h5_dset)
            data_utils.verify_book_keeping_attrs (self, h5_dset)
        os.remove(file_path)

    def test_invalid(self):
        with self.assertRaises(TypeError):
            hdf_utils.write_book_keeping_attrs(np.arange(4))


class TestPrintTreeNoMain(unittest.TestCase):

    def test_not_a_group(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)

        with h5py.File(file_path, mode='w') as h5_f:
            dset = h5_f.create_dataset('A_Dataset', data=[1, 2, 3])
            with self.assertRaises(TypeError):
                hdf_utils.print_tree(dset, rel_paths=False,
                                     main_dsets_only=False)

    def test_single_level_tree(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = ['/']
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Dataset'
            expected.append(0 * '  ' + '├ ' + obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'B_Group'
            expected.append(0 * '  ' + '├ ' + obj_name)
            expected.append((0 + 1) * '  ' + len(obj_name) * '-')
            _ = h5_f.create_group(obj_name)

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=False,
                                     main_dsets_only=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_single_level_rel_paths(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = ['/']
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Dataset'
            expected.append(obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'B_Group'
            expected.append(obj_name)
            _ = h5_f.create_group(obj_name)

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=True,
                                     main_dsets_only=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_tree(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = ['/']
        with h5py.File(file_path, mode='w') as h5_f:

            level = 0

            obj_name = 'A_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_1 = h5_f.create_group(obj_name)
            level += 1

            obj_name = 'B_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_2 = grp_1.create_group(obj_name)
            level += 1

            obj_name = 'C_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_3 = grp_2.create_group(obj_name)
            level += 1

            obj_name = 'Y_Dataset'
            expected.append(level * '  ' + '├ ' + obj_name)
            _ = grp_3.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'X_Dataset'
            expected.append(0 * '  ' + '├ ' + obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=False,
                                     main_dsets_only=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_tree_main_dsets_only(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = ['/']
        with h5py.File(file_path, mode='w') as h5_f:

            level = 0

            obj_name = 'A_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_1 = h5_f.create_group(obj_name)
            level += 1

            obj_name = 'B_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_2 = grp_1.create_group(obj_name)
            level += 1

            obj_name = 'C_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_3 = grp_2.create_group(obj_name)
            level += 1

            obj_name = 'Y_Dataset'
            # expected.append(level * '  ' + '├ ' + obj_name)
            _ = grp_3.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'X_Dataset'
            # expected.append(0 * '  ' + '├ ' + obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=False,
                                     main_dsets_only=True)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_tree_grp_a(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = []
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Group'
            grp_1 = h5_f.create_group(obj_name)
            # Full path printed for root always
            expected.append(grp_1.name)

            level = 0

            obj_name = 'B_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_2 = grp_1.create_group(obj_name)
            level += 1

            obj_name = 'C_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_3 = grp_2.create_group(obj_name)
            level += 1

            obj_name = 'Y_Dataset'
            expected.append(level * '  ' + '├ ' + obj_name)
            _ = grp_3.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'X_Dataset'
            # expected.append(0 * '  ' + '├ ' + obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(grp_1, rel_paths=False,
                                     main_dsets_only=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_tree_grp_b(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = []
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Group'
            grp_1 = h5_f.create_group(obj_name)

            obj_name = 'B_Group'
            grp_2 = grp_1.create_group(obj_name)
            # Full path printed for root always
            expected.append(grp_2.name)

            level = 0

            obj_name = 'C_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_3 = grp_2.create_group(obj_name)
            level += 1

            obj_name = 'Y_Dataset'
            expected.append(level * '  ' + '├ ' + obj_name)
            _ = grp_3.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'X_Dataset'
            # expected.append(0 * '  ' + '├ ' + obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(grp_2, rel_paths=False,
                                     main_dsets_only=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_rel_paths_grp_b(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = []
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Group'
            grp_1 = h5_f.create_group(obj_name)

            obj_name = 'B_Group'
            grp_2 = grp_1.create_group(obj_name)
            # Full path printed for root always
            expected.append(grp_2.name)

            obj_name = 'C_Group'
            grp_3 = grp_2.create_group(obj_name)
            expected.append(grp_3.name.replace(grp_2.name + '/', ''))

            obj_name = 'Y_Dataset'
            dset = grp_3.create_dataset(obj_name, data=[1, 2, 3])
            expected.append(dset.name.replace(grp_2.name + '/', ''))

            obj_name = 'X_Dataset'
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(grp_2, rel_paths=True,
                                     main_dsets_only=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_rel_paths(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = ['/']
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Group'
            grp_1 = h5_f.create_group(obj_name)
            expected.append(grp_1.name[1:])

            obj_name = 'B_Group'
            grp_2 = grp_1.create_group(obj_name)
            expected.append(grp_2.name[1:])

            obj_name = 'C_Group'
            grp_3 = grp_2.create_group(obj_name)
            expected.append(grp_3.name[1:])

            obj_name = 'Y_Dataset'
            dset = grp_3.create_dataset(obj_name, data=[1, 2, 3])
            expected.append(dset.name[1:])

            obj_name = 'X_Dataset'
            dset = h5_f.create_dataset(obj_name, data=[1, 2, 3])
            expected.append(dset.name[1:])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=True,
                                     main_dsets_only=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)


class TestPrintTreeBEPS(TestHDFUtilsBase):

    def test_root_all_dsets(self):
        level = 0
        expected = ['/',
                    level * '  ' + '├ ' + 'Raw_Measurement',
                    (level + 1) * '  ' + len('Raw_Measurement') * '-']
        level += 1
        expected += [
                    level * '  ' + '├ ' + 'Ancillary',
                    level * '  ' + '├ ' + 'Bias',
                    level * '  ' + '├ ' + 'Cycle',
                    level * '  ' + '├ ' + 'Misc',
                    (level + 1) * '  ' + len('Misc') * '-',
                    level * '  ' + '├ ' + 'Position_Indices',
                    level * '  ' + '├ ' + 'Position_Values',
                    level * '  ' + '├ ' + 'Spectroscopic_Indices',
                    level * '  ' + '├ ' + 'Spectroscopic_Values',
                    level * '  ' + '├ ' + 'X',
                    level * '  ' + '├ ' + 'Y',
                    level * '  ' + '├ ' + 'n_dim_form',
                    level * '  ' + '├ ' + 'source_main']
        level += 1
        for ind in range(2):
            expected += [
                        (level-1) * '  ' + '├ ' + 'source_main-Fitter_00'+str(ind),
                        level * '  ' + len('source_main-Fitter_000') * '-',
                        level * '  ' + '├ ' + 'Spectroscopic_Indices',
                        level * '  ' + '├ ' + 'Spectroscopic_Values',
                        level * '  ' + '├ ' + 'n_dim_form',
                        level * '  ' + '├ ' + 'results_main',
                        ]
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=False,
                                     main_dsets_only=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)

    def test_root_main_dsets_only(self):
        level = 0
        expected = ['/',
                    level * '  ' + '├ ' + 'Raw_Measurement',
                    (level + 1) * '  ' + len('Raw_Measurement') * '-']
        level += 1
        expected += [
            level * '  ' + '├ ' + 'Misc',
            (level + 1) * '  ' + len('Misc') * '-',
            level * '  ' + '├ ' + 'source_main']
        level += 1
        for ind in range(2):
            expected += [
                (level - 1) * '  ' + '├ ' + 'source_main-Fitter_00' + str(ind),
                level * '  ' + len('source_main-Fitter_000') * '-',
                level * '  ' + '├ ' + 'results_main',
            ]
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=False,
                                     main_dsets_only=True)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
