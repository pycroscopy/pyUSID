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
import shutil

sys.path.append("../../pyUSID/")
from pyUSID.io import hdf_utils, write_utils, USIDataset

from .. import data_utils

if sys.version_info.major == 3:
    unicode = str


class TestSimple(unittest.TestCase):

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

    def test_copy_region_refs(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(11, 7)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_dest = h5_f.create_dataset('Target', data=data)
            source_ref = h5_dset_source.regionref[0:-1:2]
            h5_dset_source.attrs['regref'] = source_ref

            hdf_utils.copy_region_refs(h5_dset_source, h5_dset_dest)

            self.assertTrue(np.allclose(h5_dset_source[h5_dset_source.attrs['regref']],
                                        h5_dset_dest[h5_dset_dest.attrs['regref']]))

        os.remove(file_path)


class TestCheckIfMain(TestSimple):

    def test_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/results_main']]
            for dset in expected_dsets:
                self.assertTrue(hdf_utils.check_if_main(dset, verbose=False))

    def test_illegal_01(self):
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

    def test_anc_not_dsets(self):
        temp_path = 'test.h5'
        data_utils.delete_existing_file(temp_path)
        with h5py.File(temp_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=np.random.rand(2, 3))
            for anc_dset_name in ['Position_Indices', 'Position_Values',
                                  'Spectroscopic_Indices', 'Spectroscopic_Values']:
                h5_dset.attrs[anc_dset_name] = h5_f.ref
            self.assertFalse(hdf_utils.check_if_main(h5_dset, verbose=False))
        os.remove(temp_path)

    def test_missing_str_attrs(self):
        temp_path = 'test.h5'
        data_utils.delete_existing_file(temp_path)
        with h5py.File(temp_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=np.random.rand(2, 3))
            for anc_dset_name in ['Position_Indices', 'Position_Values',
                                  'Spectroscopic_Indices', 'Spectroscopic_Values']:
                h5_dset.attrs[anc_dset_name] = h5_dset.ref
            self.assertFalse(hdf_utils.check_if_main(h5_dset, verbose=False))
        os.remove(temp_path)

    def test_invalid_str_attrs(self):
        temp_path = 'test.h5'
        data_utils.delete_existing_file(temp_path)
        with h5py.File(temp_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=np.random.rand(2, 3))
            h5_dset.attrs['quantity'] = [1, 2, 3]
            h5_dset.attrs['units'] = 4.1234
            for anc_dset_name in ['Position_Indices', 'Position_Values',
                                  'Spectroscopic_Indices', 'Spectroscopic_Values']:
                h5_dset.attrs[anc_dset_name] = h5_dset.ref
            self.assertFalse(hdf_utils.check_if_main(h5_dset, verbose=False))
        os.remove(temp_path)

    def test_anc_shapes_not_matching(self):
        temp_path = 'test.h5'
        data_utils.delete_existing_file(temp_path)
        with h5py.File(temp_path, mode='w') as h5_f:
            h5_main = h5_f.create_dataset('Main', data=np.random.rand(2, 3))
            h5_pos_ind = h5_f.create_dataset('Pos_Inds', data=np.random.rand(2, 1))
            h5_spec_ind = h5_f.create_dataset('Spec_Inds', data=np.random.rand(1, 5))
            h5_main.attrs['quantity'] = 'quant'
            h5_main.attrs['units'] = 'unit'
            for anc_dset_name in ['Position_Indices', 'Position_Values']:
                h5_main.attrs[anc_dset_name] = h5_pos_ind.ref
            for anc_dset_name in ['Spectroscopic_Indices', 'Spectroscopic_Values']:
                h5_main.attrs[anc_dset_name] = h5_spec_ind.ref
            self.assertFalse(hdf_utils.check_if_main(h5_main, verbose=False))
        os.remove(temp_path)


class TestGetSourceDataset(TestSimple):

    def test_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_groups = [h5_f['/Raw_Measurement/source_main-Fitter_000'],
                        h5_f['/Raw_Measurement/source_main-Fitter_001']]
            h5_main = USIDataset(h5_f['/Raw_Measurement/source_main'])
            for h5_grp in h5_groups:
                self.assertEqual(h5_main, hdf_utils.get_source_dataset(h5_grp))

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.get_source_dataset('/Raw_Measurement/Misc')

    def test_illegal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(ValueError):
                _ = hdf_utils.get_source_dataset(h5_f['/Raw_Measurement/Misc'])


class TestGetAllMain(TestSimple):

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.get_all_main("sdsdsds")

        with self.assertRaises(TypeError):
            _ = hdf_utils.get_all_main(np.arange(4))

    def test_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/results_main']]
            main_dsets = hdf_utils.get_all_main(h5_f, verbose=False)
            # self.assertEqual(set(main_dsets), set(expected_dsets))
            self.assertEqual(len(main_dsets), len(expected_dsets))
            self.assertTrue(np.all([x.name == y.name for x, y in zip(main_dsets, expected_dsets)]))


class TestWriteIndValDsets(TestSimple):

    def base_bare_minimum_inputs(self, slow_to_fast, is_spectral):
        num_cols = 3
        num_rows = 2
        sizes = [num_cols, num_rows]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        if slow_to_fast:
            dim_names = dim_names[::-1]
            dim_units = dim_units[::-1]
            sizes = sizes[::-1]

        descriptor = []
        for length, name, units in zip(sizes, dim_names, dim_units):
            descriptor.append(write_utils.Dimension(name, units, np.arange(length)))

        inds_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols)))\

        if not is_spectral:
            inds_data = inds_data.T

        if slow_to_fast:
            func = np.flipud if is_spectral else np.fliplr
            inds_data = func(inds_data)

        file_path = 'test_write_ind_val_dsets.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_inds, h5_vals = hdf_utils.write_ind_val_dsets(h5_f, descriptor, is_spectral=is_spectral,
                                                             slow_to_fast=slow_to_fast)

            data_utils.validate_aux_dset_pair(self, h5_f, h5_inds, h5_vals, dim_names, dim_units, inds_data,
                                              is_spectral=is_spectral, slow_to_fast=slow_to_fast)

        os.remove(file_path)

    def test_legal_bare_minimum_pos_f2s(self):
        self.base_bare_minimum_inputs(False, False)

    def test_legal_bare_minimum_pos_s2f(self):
        self.base_bare_minimum_inputs(True, False)

    def test_legal_bare_minimum_spec_f2s(self):
        self.base_bare_minimum_inputs(False, True)

    def test_legal_bare_minimum_spec_s2f(self):
        self.base_bare_minimum_inputs(True, True)

    def test_legal_override_steps_offsets_base_name(self):
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
        # Sending in Fast to Slow but what comes out is slow to fast
        spec_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols)))
        spec_vals = np.vstack((np.tile(np.arange(num_cols), num_rows) * col_step + col_initial,
                              np.repeat(np.arange(num_rows), num_cols) * row_step + row_initial))

        file_path = 'test_write_ind_val_dsets.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group("Blah")
            h5_inds, h5_vals = hdf_utils.write_ind_val_dsets(h5_group, descriptor, is_spectral=True,
                                                             base_name=new_base_name, slow_to_fast=False)
            data_utils.validate_aux_dset_pair(self, h5_group, h5_inds, h5_vals, dim_names, dim_units, spec_inds,
                                          vals_matrix=spec_vals, base_name=new_base_name, is_spectral=True, slow_to_fast=False)
        os.remove(file_path)

    def test_illegal(self):
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


class TestWriteReducedAncDsets(TestSimple):

    def test_spec_2d_to_1d(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)
        shutil.copy(data_utils.std_beps_path, duplicate_path)
        with h5py.File(duplicate_path, mode='r+') as h5_f:
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

    def test_spec_2d_to_1d_new_file(self):
        new_file_path = 'reduced_ancs.h5'
        data_utils.delete_existing_file(new_file_path)
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_spec_inds_orig = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_spec_vals_orig = h5_f['/Raw_Measurement/Spectroscopic_Values']
            new_base_name = 'Blah'
            with h5py.File(new_file_path, mode='w') as h5_f_new:

                h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_anc_dsets(h5_f_new,
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
                    self.assertEqual(h5_dset.parent, h5_f_new)
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

        os.remove(new_file_path)

    def test_spec_1d_to_0d(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path, mode='w') as h5_f:

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

    def test_3d_to_1d_pos_fastest_n_slowest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path, mode='w') as h5_f:

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

    def test_3d_to_1d_spec_fastest_n_slowest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path, mode='w') as h5_f:

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

    def test_3d_to_1d_spec_fastest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path, mode='w') as h5_f:

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

    def test_3d_to_1d_spec_slowest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path, mode='w') as h5_f:

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

    def test_3d_to_2d_spec_fastest_n_slowest(self):
        duplicate_path = 'copy_test_hdf_utils.h5'
        data_utils.delete_existing_file(duplicate_path)

        with h5py.File(duplicate_path, mode='w') as h5_f:

            dims = [write_utils.Dimension('Freq', 'Hz', np.linspace(300, 350, 5)),
                    write_utils.Dimension('Bias', 'V', [-2, 4, 10]),
                    write_utils.Dimension('Cycle', 'a.u.', 2)]

            h5_spec_inds_orig, h5_spec_vals_orig = hdf_utils.write_ind_val_dsets(h5_f, dims, is_spectral=True,
                                                                                 slow_to_fast=False)
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

            # Sending in Fast to Slow but what comes out is slow to fast
            ref_inds = np.flipud(ref_inds)
            ref_vals = np.flipud(ref_vals)
            dim_names = dim_names[::-1]
            dim_units = dim_units[::-1]

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


class TestFindDataset(TestSimple):

    def test_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/']
            with self.assertRaises(TypeError):
                ret_val = hdf_utils.find_dataset(h5_group, np.arange(4))

    def test_invalid_type_dset(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.find_dataset(4.324, 'Spectroscopic_Indices')

    def test_illegal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/']
            ret_val = hdf_utils.find_dataset(h5_group, 'Does_Not_Exist')
            self.assertEqual(len(ret_val), 0)


class TestFindResultsGroup(TestSimple):

    def test_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected_groups = [h5_f['/Raw_Measurement/source_main-Fitter_000'],
                               h5_f['/Raw_Measurement/source_main-Fitter_001']]
            ret_val = hdf_utils.find_results_groups(h5_main, 'Fitter')
            self.assertEqual(set(ret_val), set(expected_groups))

    def test_no_dset(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                _ = hdf_utils.find_results_groups(h5_f, 'Fitter')

    def test_not_string(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(TypeError):
                _ = hdf_utils.find_results_groups(h5_main, np.arange(5))

    def test_no_such_tool(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            ret_val = hdf_utils.find_results_groups(h5_main, 'Blah')
            self.assertEqual(len(ret_val), 0)

    def test_results_in_diff_file(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)

        new_path = 'new.h5'
        data_utils.delete_existing_file(new_path)

        with h5py.File(file_path, mode='w') as h5_f:
            h5_main = h5_f.create_dataset('Main', data=[1, 2, 3])
            with h5py.File(new_path, mode='w') as h5_f_2:
                grp_1 = h5_f_2.create_group('Main-Tool_000')
                grp_2 = h5_f_2.create_group('Main-Tool_001')
                grps = hdf_utils.find_results_groups(h5_main, 'Tool',
                                                     h5_parent_group=h5_f_2)
                self.assertEqual(set([grp_1, grp_2]), set(grps))

        os.remove(file_path)
        os.remove(new_path)

    def test_results_in_diff_file_invalid_type(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_main = h5_f.create_dataset('Main', data=[1, 2, 3])
            with self.assertRaises(TypeError):
                _ = hdf_utils.find_results_groups(h5_main, 'Tool',
                                                  h5_parent_group=h5_main)

        os.remove(file_path)


class TestCheckForMatchingAttrs(TestSimple):

    def test_dset_no_attrs(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            self.assertTrue(hdf_utils.check_for_matching_attrs(h5_main, new_parms=None))

    def test_dset_matching_attrs(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': 'A', 'quantity':'Current'}
            self.assertTrue(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_dset_one_mismatched_attrs(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': 'A', 'blah': 'meh'}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            self.assertTrue(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp_mismatched_types_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': 'string_val'}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp_mismatched_types_02(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_1': ['str_1', 'str_2', 'str_3']}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp_mismatched_types_03(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': [1, 4.234, 'str_3']}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp_mismatched_types_04(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': [1, 4.234, 45]}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))


class TestCheckForOld(TestSimple):

    def test_invalid_types(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(TypeError):
                _ = hdf_utils.check_for_old("h5_main", "blah")

            with self.assertRaises(TypeError):
                _ = hdf_utils.check_for_old(np.arange(4), "blah")

            with self.assertRaises(TypeError):
                _ = hdf_utils.check_for_old(h5_main, 1.234)

            with self.assertRaises(TypeError):
                _ = hdf_utils.check_for_old(h5_main, 'Fitter',
                                            new_parms="not_a_dictionary")

            with self.assertRaises(TypeError):
                _ = hdf_utils.check_for_old(h5_main, 'Fitter',
                                            target_dset=1.234)

    def test_valid_target_dset(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': ['V'], 'labels': ['Bias']}
            dset_name = 'Spectroscopic_Indices'
            groups = hdf_utils.check_for_old(h5_main, 'Fitter',
                                             new_parms=attrs,
                                             target_dset=dset_name,
                                             verbose=False)
            groups = set(groups)
            self.assertEqual(groups, set([h5_f['/Raw_Measurement/source_main-Fitter_000/'],
                                          h5_f['/Raw_Measurement/source_main-Fitter_001/']]))

    def test_invalid_target_dset(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2',
                                                      'str_3']}
            ret = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs,
                                          target_dset='Does_not_exist')
            self.assertEqual(ret, [])

    def test_exact_match(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            [h5_ret_grp] = hdf_utils.check_for_old(h5_main, 'Fitter',
                                                   new_parms=attrs,
                                                   target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_000'])

    def test_subset_but_match(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            [h5_ret_grp] = hdf_utils.check_for_old(h5_main, 'Fitter',
                                                   new_parms=attrs,
                                                   target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_000'])

    def test_exact_match_02(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'other_string_val', 'att_2': 5.4321,
                     'att_3': [4, 1, 3], 'att_4': ['s', 'str_2', 'str_3']}
            [h5_ret_grp] = hdf_utils.check_for_old(h5_main, 'Fitter',
                                                   new_parms=attrs,
                                                   target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_001'])

    def test_fail_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': [4, 1, 3], 'att_2': ['s', 'str_2', 'str_3'],
                     'att_3': 'other_string_val', 'att_4': 5.4321}
            ret_val = hdf_utils.check_for_old(h5_main, 'Fitter',
                                              new_parms=attrs, target_dset=None)
            self.assertIsInstance(ret_val, list)
            self.assertEqual(len(ret_val), 0)

    def test_fail_02(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_x': [4, 1, 3], 'att_z': ['s', 'str_2', 'str_3'],
                     'att_y': 'other_string_val', 'att_4': 5.4321}
            ret_val = hdf_utils.check_for_old(h5_main, 'Fitter',
                                              new_parms=attrs, target_dset=None)
            self.assertIsInstance(ret_val, list)
            self.assertEqual(len(ret_val), 0)


class TestCreateIndexedGroup(unittest.TestCase):

    def test_first_group(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = hdf_utils.create_indexed_group(h5_f, 'Hello')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Hello_000')
            self.assertEqual(h5_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group)

            h5_sub_group = hdf_utils.create_indexed_group(h5_group, 'Test')
            self.assertIsInstance(h5_sub_group, h5py.Group)
            self.assertEqual(h5_sub_group.name, '/Hello_000/Test_000')
            self.assertEqual(h5_sub_group.parent, h5_group)
            data_utils.verify_book_keeping_attrs(self, h5_sub_group)
        os.remove(file_path)

    def test_second(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group_1 = hdf_utils.create_indexed_group(h5_f, 'Hello')
            self.assertIsInstance(h5_group_1, h5py.Group)
            self.assertEqual(h5_group_1.name, '/Hello_000')
            self.assertEqual(h5_group_1.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group_1)

            h5_group_2 = hdf_utils.create_indexed_group(h5_f, 'Hello')
            self.assertIsInstance(h5_group_2, h5py.Group)
            self.assertEqual(h5_group_2.name, '/Hello_001')
            self.assertEqual(h5_group_2.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group_2)
        os.remove(file_path)

    def test_w_suffix_(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = hdf_utils.create_indexed_group(h5_f, 'Hello_')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Hello_000')
            self.assertEqual(h5_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group)
        os.remove(file_path)

    def test_empty_base_name(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            with self.assertRaises(ValueError):
                _ = hdf_utils.create_indexed_group(h5_f, '    ')
        os.remove(file_path)

    def test_create_indexed_group_invalid_types(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.create_indexed_group(np.arange(4), "fddfd")

        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            with self.assertRaises(TypeError):
                _ = hdf_utils.create_indexed_group(h5_f, 1.2343)
        os.remove(file_path)


class TestCreateResultsGroup(unittest.TestCase):

    def test_first(self):
        self.helper_first()

    def test_dash_in_name(self):
        self.helper_first(add_dash_to_name=True)

    def helper_first(self, add_dash_to_name=False):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            if add_dash_to_name:
                h5_group = hdf_utils.create_results_group(h5_dset, 'Some-Tool')
                tool_name = 'Some_Tool'
            else:
                tool_name = 'Tool'
                h5_group = hdf_utils.create_results_group(h5_dset, tool_name)
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Main-' + tool_name + '_000')
            self.assertEqual(h5_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group)

            h5_dset = h5_group.create_dataset('Main_Dataset', data=[1, 2, 3])
            h5_sub_group = hdf_utils.create_results_group(h5_dset, 'SHO_Fit')
            self.assertIsInstance(h5_sub_group, h5py.Group)
            self.assertEqual(h5_sub_group.name, '/Main-' + tool_name + '_000/Main_Dataset-SHO_Fit_000')
            self.assertEqual(h5_sub_group.parent, h5_group)
            data_utils.verify_book_keeping_attrs(self, h5_sub_group)
        os.remove(file_path)

    def test_second(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            h5_group = hdf_utils.create_results_group(h5_dset, 'Tool')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Main-Tool_000')
            self.assertEqual(h5_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group)

            h5_sub_group = hdf_utils.create_results_group(h5_dset, 'Tool')
            self.assertIsInstance(h5_sub_group, h5py.Group)
            self.assertEqual(h5_sub_group.name, '/Main-Tool_001')
            self.assertEqual(h5_sub_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_sub_group)
        os.remove(file_path)

    def test_empty_tool_name(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            with self.assertRaises(ValueError):
                _ = hdf_utils.create_results_group(h5_dset, '   ')
        os.remove(file_path)

    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.create_results_group("not a dataset", 'Tool')

        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            with self.assertRaises(TypeError):
                _ = hdf_utils.create_results_group(h5_f, 'Tool')

            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            with self.assertRaises(TypeError):
                _ = hdf_utils.create_results_group(h5_dset, 'Tool',
                                                   h5_parent_group='not_group')

        os.remove(file_path)

    def test_different_file(self):
        file_path = 'test.h5'
        new_path = 'new.h5'
        data_utils.delete_existing_file(file_path)
        data_utils.delete_existing_file(new_path)

        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            # Ensuring that index is calculated at destination, not source:
            _ = h5_f.create_group('Main-Tool_000')

            with h5py.File(new_path, mode='w') as h5_f_new:
                _ = h5_f_new.create_group('Main-Tool_000')

                h5_group = hdf_utils.create_results_group(h5_dset, 'Tool',
                                                          h5_parent_group=h5_f_new)

                self.assertIsInstance(h5_group, h5py.Group)
                self.assertEqual(h5_group.name, '/Main-Tool_001')
                self.assertEqual(h5_group.parent, h5_f_new)
                self.assertNotEqual(h5_dset.file, h5_group.file)
                data_utils.verify_book_keeping_attrs(self, h5_group)

        os.remove(file_path)
        os.remove(new_path)


class TestAssignGroupIndex(TestSimple):

    def test_existing(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            ret_val = hdf_utils.assign_group_index(h5_group, 'source_main-Fitter')
            self.assertEqual(ret_val, 'source_main-Fitter_002')

    def test_new(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            ret_val = hdf_utils.assign_group_index(h5_group, 'blah_')
            self.assertEqual(ret_val, 'blah_000')

    def test_invalid_dtypes(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.assign_group_index("not a dataset", 'blah_')

        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            with self.assertRaises(TypeError):
                _ = hdf_utils.assign_group_index(h5_group, 1.24)


class TestLinkAsMain(unittest.TestCase):

    def test_pos_args_not_h5_dset(self):
        file_path = 'link_as_main.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset("Blah", data=np.random.rand(2, 3))
            with self.assertRaises(TypeError):
                hdf_utils.link_as_main("h5_main", 1.234, -2, False, {"h5_spec_vals": 2.432})

            with self.assertRaises(TypeError):
                hdf_utils.link_as_main(h5_dset, 1.234, -2, False, {"h5_spec_vals": 2.432})

            with self.assertRaises(TypeError):
                hdf_utils.link_as_main(h5_dset, h5_dset, -2, False, {"h5_spec_vals": 2.432})

            with self.assertRaises(TypeError):
                hdf_utils.link_as_main(h5_dset, h5_dset, h5_dset, False, {"h5_spec_vals": 2.432})

            with self.assertRaises(TypeError):
                hdf_utils.link_as_main(h5_dset, h5_dset, h5_dset, h5_dset, {"h5_spec_vals": 2.432})

        data_utils.delete_existing_file(file_path)

    def test_anc_args_not_h5_dset(self):
        file_path = 'link_as_main.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset("Blah", data=np.random.rand(2, 3))
            with self.assertRaises(TypeError):
                hdf_utils.link_as_main("h5_main", 1.234, -2, False, {"h5_spec_vals": 2.432})

            with self.assertRaises(TypeError):
                hdf_utils.link_as_main(h5_dset, 1.234, -2, False, {"h5_spec_vals": 2.432})

            with self.assertRaises(TypeError):
                hdf_utils.link_as_main(h5_dset, h5_dset, -2, False, {"h5_spec_vals": 2.432})

            with self.assertRaises(TypeError):
                hdf_utils.link_as_main(h5_dset, h5_dset, h5_dset, False, {"h5_spec_vals": 2.432})

            with self.assertRaises(TypeError):
                hdf_utils.link_as_main(h5_dset, h5_dset, h5_dset, h5_dset, {"h5_spec_vals": 2.432})

        data_utils.delete_existing_file(file_path)

    def test_ind_vals_not_same_shape(self):
        file_path = 'link_as_main.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_main = h5_f.create_dataset("Blah", data=np.zeros((3, 5), dtype=np.uint16))
            h5_pos_inds = h5_f.create_dataset("P_I", data=np.zeros((7, 2), dtype=np.uint16))
            h5_pos_vals = h5_f.create_dataset("P_V", data=np.zeros((3, 2), dtype=np.uint16))
            h5_spec_inds = h5_f.create_dataset("S_I", data=np.zeros((2, 5), dtype=np.uint16))
            h5_spec_vals = h5_f.create_dataset("S_V", data=np.zeros((2, 5), dtype=np.uint16))
            with self.assertRaises(ValueError):
                hdf_utils.link_as_main(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)

        data_utils.delete_existing_file(file_path)

        with h5py.File(file_path, mode='w') as h5_f:
            h5_main = h5_f.create_dataset("Blah", data=np.zeros((3, 5), dtype=np.uint16))
            h5_pos_inds = h5_f.create_dataset("P_I", data=np.zeros((3, 2), dtype=np.uint16))
            h5_pos_vals = h5_f.create_dataset("P_V", data=np.zeros((3, 2), dtype=np.uint16))
            h5_spec_inds = h5_f.create_dataset("S_I", data=np.zeros((2, 8), dtype=np.uint16))
            h5_spec_vals = h5_f.create_dataset("S_V", data=np.zeros((2, 5), dtype=np.uint16))
            with self.assertRaises(ValueError):
                hdf_utils.link_as_main(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)

        data_utils.delete_existing_file(file_path)

    def helper_test(self, quant_units_specified):
        file_path = 'link_as_main.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_raw_grp = h5_f.create_group('Raw_Measurement')

            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_dset_name = 'source_main'

            source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

            h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices', data=source_pos_data, dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_inds, pos_attrs)

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            data_utils.write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''], 'labels': ['Bias', 'Cycle']}

            h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                            dtype=np.uint16)
            data_utils.write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

            h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                            dtype=np.float32)
            data_utils.write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
            data_utils.write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)

            expected_type = h5py.Dataset

            if quant_units_specified:
                expected_type = USIDataset
                data_utils.write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})

            self.assertFalse(hdf_utils.check_if_main(h5_source_main))

            # Now need to link as main!
            usid_source = hdf_utils.link_as_main(h5_source_main, h5_pos_inds, h5_pos_vals, h5_source_spec_inds,
                                                 h5_source_spec_vals)

            # Finally:
            if quant_units_specified:
                self.assertTrue(hdf_utils.check_if_main(h5_source_main))

            self.assertIsInstance(usid_source, expected_type)

        os.remove(file_path)

    def test_typical_attrs_specified(self):
        self.helper_test(True)

    def test_typical_attrs_not_specified(self):
        self.helper_test(False)


class TestCopyAttributes(unittest.TestCase):

    def test_not_h5_dset(self):
        temp_path = 'copy_attributes.h5'
        with h5py.File(temp_path, mode='w') as h5_f:
            h5_grp = h5_f.create_group('Blah')
            with self.assertRaises(TypeError):
                hdf_utils.copy_attributes(h5_grp, np.arange(4))

            with self.assertRaises(TypeError):
                hdf_utils.copy_attributes(np.arange(4), h5_grp)
        os.remove(temp_path)

    def test_file_dset(self):
        file_path = 'test.h5'
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        also_easy_attr = {'N_numbers': [1, -53.6, 0.000463]}
        hard_attrs = {'N_strings': np.array(['a', 'bc', 'def'], dtype='S')}
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_f.attrs.update(easy_attrs)
            h5_f.attrs.update(also_easy_attr)
            h5_f.attrs.update(hard_attrs)
            h5_dset = h5_f.create_dataset('Main_01', data=[1, 2, 3])
            hdf_utils.copy_attributes(h5_f, h5_dset)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset.attrs[key])
            for key, val in also_easy_attr.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
            for key, val in hard_attrs.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
        os.remove(file_path)

    def test_group_dset(self):
        file_path = 'test.h5'
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        also_easy_attr = {'N_numbers': [1, -53.6, 0.000463]}
        hard_attrs = {'N_strings': np.array(['a', 'bc', 'def'], dtype='S')}
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group('Group')
            h5_group.attrs.update(easy_attrs)
            h5_group.attrs.update(also_easy_attr)
            h5_group.attrs.update(hard_attrs)
            h5_dset = h5_f.create_dataset('Main_01', data=[1, 2, 3])
            hdf_utils.copy_attributes(h5_group, h5_dset)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset.attrs[key])
            for key, val in also_easy_attr.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
            for key, val in hard_attrs.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
        os.remove(file_path)

    def test_dset_w_reg_ref_group_but_skipped(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_source.attrs.update(easy_attrs)
            h5_dset_sink = h5_f.create_dataset('Sink', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}
            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            hdf_utils.copy_attributes(h5_dset_source, h5_dset_sink, skip_refs=True)

            self.assertEqual(len(h5_dset_sink.attrs), len(easy_attrs))
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset_sink.attrs[key])

        os.remove(file_path)

    def test_dset_w_reg_ref_group_to_file(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_source.attrs.update(easy_attrs)
            reg_refs = {'even_rows': (slice(None), slice(0, None, 2)),
                        'odd_rows': (slice(None), slice(1, None, 2))}
            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    hdf_utils.copy_attributes(h5_dset_source, h5_f, skip_refs=False)
            else:
                hdf_utils.copy_attributes(h5_dset_source, h5_f, skip_refs=False)

            self.assertEqual(len(h5_f.attrs), len(easy_attrs))
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_f.attrs[key])

        os.remove(file_path)

    def test_dset_w_reg_ref_group(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_source.attrs.update(easy_attrs)
            h5_dset_sink = h5_f.create_dataset('Sink', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}
            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            hdf_utils.copy_attributes(h5_dset_source, h5_dset_sink, skip_refs=False)

            self.assertEqual(len(h5_dset_sink.attrs), len(reg_refs) + len(easy_attrs))
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset_sink.attrs[key])

            self.assertTrue('labels' not in h5_dset_sink.attrs.keys())

            expected_data = [data[0:None:2, :], data[1:None:2, :]]
            written_data = [h5_dset_sink[h5_dset_sink.attrs['even_rows']],
                            h5_dset_sink[h5_dset_sink.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_illegal_to_from_reg_ref(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_dest = h5_f.create_dataset('Sink', data=data[:-1, :-1])
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}
            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    hdf_utils.copy_attributes(h5_dset_source, h5_dset_dest, skip_refs=False)
            else:
                hdf_utils.copy_attributes(h5_dset_source, h5_dset_dest, skip_refs=False)


class TestCopyMainAttributes(unittest.TestCase):

    def test_valid(self):
        file_path = 'test.h5'
        main_attrs = {'quantity': 'Current', 'units': 'nA'}
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Main_01', data=[1, 23])
            h5_dset_source.attrs.update(main_attrs)
            h5_group = h5_f.create_group('Group')
            h5_dset_sink = h5_group.create_dataset('Main_02', data=[4, 5])
            hdf_utils.copy_main_attributes(h5_dset_source, h5_dset_sink)
            for key, val in main_attrs.items():
                self.assertEqual(val, h5_dset_sink.attrs[key])
        os.remove(file_path)

    def test_no_main_attrs(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Main_01', data=[1, 23])
            h5_group = h5_f.create_group('Group')
            h5_dset_sink = h5_group.create_dataset('Main_02', data=[4, 5])
            with self.assertRaises(KeyError):
                hdf_utils.copy_main_attributes(h5_dset_source, h5_dset_sink)
        os.remove(file_path)

    def test_wrong_objects(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Main_01', data=[1, 23])
            h5_group = h5_f.create_group('Group')
            with self.assertRaises(TypeError):
                hdf_utils.copy_main_attributes(h5_dset_source, h5_group)
            with self.assertRaises(TypeError):
                hdf_utils.copy_main_attributes(h5_group, h5_dset_source)
        os.remove(file_path)


class TestCreateEmptyDataset(unittest.TestCase):

    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.create_empty_dataset("not a dataset", np.float16, 'Duplicate')

        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            with self.assertRaises(TypeError):
                _ = hdf_utils.create_empty_dataset(h5_dset_source, np.arange(15), 'Duplicate')

            with self.assertRaises(TypeError):
                _ = hdf_utils.create_empty_dataset(h5_dset_source, np.float32, {'not a': 'string'})

            with self.assertRaises(TypeError):
                _ = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Duplicate',
                                                   new_attrs="not_a_dictionary")

            with self.assertRaises(TypeError):
                _ = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Duplicate',
                                                   h5_group=h5_dset_source)

        os.remove(file_path)

    def test_same_group_new_attrs(self):
        file_path = 'test.h5'
        existing_attrs = {'a': 1, 'b': 'Hello'}
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_source.attrs.update(existing_attrs)
            h5_duplicate = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Duplicate', new_attrs=easy_attrs)
            self.assertIsInstance(h5_duplicate, h5py.Dataset)
            self.assertEqual(h5_duplicate.parent, h5_dset_source.parent)
            self.assertEqual(h5_duplicate.name, '/Duplicate')
            self.assertEqual(h5_duplicate.dtype, np.float16)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_duplicate.attrs[key])
            for key, val in existing_attrs.items():
                self.assertEqual(val, h5_duplicate.attrs[key])

        os.remove(file_path)

    def validate_copied_dataset(self, h5_f_new, h5_dest, dset_new_name,
                                dset_data, dset_attrs):
        self.assertTrue(dset_new_name in h5_f_new.keys())
        h5_anc_dest = h5_f_new[dset_new_name]
        self.assertIsInstance(h5_anc_dest, h5py.Dataset)
        self.assertTrue(np.allclose(dset_data, h5_anc_dest[()]))
        self.assertEqual(len(dset_attrs),
                         len(h5_anc_dest.attrs.keys()))
        for key, val in dset_attrs.items():
            self.assertEqual(val, h5_anc_dest.attrs[key])
        self.assertTrue(dset_new_name in h5_dest.attrs.keys())
        self.assertEqual(h5_f_new[h5_dest.attrs[dset_new_name]],
                         h5_anc_dest)

    def test_diff_file_new_attrs_linked_dsets(self):
        file_path = 'test.h5'
        new_path = 'new.h5'
        existing_attrs = {'a': 1, 'b': 'Hello'}
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_source.attrs.update(existing_attrs)
            h5_other = h5_f.create_dataset('Other', data=[1, 2, 3])
            anc_attrs = {'a': 1, 'b': -55}
            h5_other.attrs.update(anc_attrs)
            h5_dset_source.attrs['dset_ref'] = h5_other.ref
            with h5py.File(new_path, mode='w') as h5_f_new:
                func = hdf_utils.create_empty_dataset
                if sys.version_info.major == 3:
                    with self.assertWarns(UserWarning):
                        h5_duplicate = func(h5_dset_source, np.float16,
                                            'Duplicate', h5_group=h5_f_new,
                                            new_attrs=easy_attrs,
                                            skip_refs=False)
                else:
                    h5_duplicate = func(h5_dset_source, np.float16,
                                        'Duplicate', h5_group=h5_f_new,
                                        new_attrs=easy_attrs, skip_refs=False)
                self.assertIsInstance(h5_duplicate, h5py.Dataset)
                self.assertEqual(h5_duplicate.parent, h5_f_new)
                self.assertFalse(h5_dset_source.file == h5_duplicate.file)
                self.assertEqual(h5_duplicate.name, '/Duplicate')
                self.assertEqual(h5_duplicate.dtype, np.float16)
                for key, val in easy_attrs.items():
                    self.assertEqual(val, h5_duplicate.attrs[key])
                for key, val in existing_attrs.items():
                    self.assertEqual(val, h5_duplicate.attrs[key])
                self.assertTrue('dset_ref' in h5_duplicate.attrs.keys())
                self.validate_copied_dataset(h5_f_new, h5_duplicate,
                                             'dset_ref', h5_other[()],
                                             anc_attrs)

        os.remove(file_path)

    def test_diff_groups(self):
        file_path = 'test.h5'
        existing_attrs = {'a': 1, 'b': 'Hello'}
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_source.attrs.update(existing_attrs)
            h5_group = h5_f.create_group('Group')
            h5_duplicate = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Duplicate',
                                                          h5_group=h5_group, new_attrs=easy_attrs)
            self.assertIsInstance(h5_duplicate, h5py.Dataset)
            self.assertEqual(h5_duplicate.parent, h5_group)
            self.assertEqual(h5_duplicate.name, '/Group/Duplicate')
            self.assertEqual(h5_duplicate.dtype, np.float16)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_duplicate.attrs[key])
            for key, val in existing_attrs.items():
                self.assertEqual(val, h5_duplicate.attrs[key])

        os.remove(file_path)

    def test_w_region_refs(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        main_attrs = {'quantity': 'Current', 'units': 'nA'}
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_source.attrs.update(main_attrs)
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}

            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            h5_copy = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Existing')

            for reg_ref_name in reg_refs.keys():
                self.assertTrue(isinstance(h5_copy.attrs[reg_ref_name], h5py.RegionReference))
                self.assertTrue(h5_dset_source[h5_dset_source.attrs[reg_ref_name]].shape == h5_copy[
                    h5_copy.attrs[reg_ref_name]].shape)

        os.remove(file_path)

    def test_existing_dset_name(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            _ = h5_f.create_dataset('Existing', data=[4, 5, 6])
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    h5_duplicate = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Existing')
            else:
                h5_duplicate = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Existing')
            self.assertIsInstance(h5_duplicate, h5py.Dataset)
            self.assertEqual(h5_duplicate.name, '/Existing')
            self.assertTrue(np.allclose(h5_duplicate[()], np.zeros(3)))
            self.assertEqual(h5_duplicate.dtype, np.float16)
        os.remove(file_path)


class TestCheckAndLinkAncillary(TestSimple):
    
    def test_not_dset(self):
        with self.assertRaises(TypeError):
            hdf_utils.check_and_link_ancillary(np.arange(5), ['Spec'])

    def test_h5_main_not_dset(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            with self.assertRaises(TypeError):
                hdf_utils.check_and_link_ancillary(h5_dset_source, ['Spec'],
                                                   h5_main="not_a_dataset")
        os.remove(file_path)

    def test_one_dset_to_name(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_2 = h5_f.create_dataset('Other', data=[1, 2, 3])
            att_name = 'root'
            expected = h5_dset_2
            hdf_utils.check_and_link_ancillary(h5_dset_source, att_name,
                                               h5_main=None, anc_refs=expected)
            # Only one attribute expected
            self.assertEqual(len(h5_dset_source.attrs.keys()), 1)
            self.assertTrue(att_name in h5_dset_source.attrs.keys())
            actual = h5_dset_source.attrs[att_name]
            self.assertIsInstance(actual, h5py.Reference)
            self.assertEqual(h5_f[actual], expected)
        os.remove(file_path)

    def test_many_objs_to_many_names(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_2 = h5_f.create_dataset('Other', data=[1, 2, 3])
            h5_grp = h5_f.create_group('Blah')
            att_names = ['Meh', 'Wah']
            expected = [h5_dset_2, h5_grp]
            hdf_utils.check_and_link_ancillary(h5_dset_source, att_names,
                                               h5_main=None, anc_refs=expected)
            self.assertEqual(len(h5_dset_source.attrs.keys()), len(att_names))
            self.assertEqual(set(att_names), set(h5_dset_source.attrs.keys()))
            for name, exp_val in zip(att_names, expected):
                actual = h5_dset_source.attrs[name]
                self.assertIsInstance(actual, h5py.Reference)
                self.assertEqual(h5_f[actual], exp_val)
        os.remove(file_path)

    def test_objs_and_refs_to_names(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_2 = h5_f.create_dataset('Other', data=[1, 2, 3])
            h5_grp = h5_f.create_group('Blah')
            att_names = ['Meh', 'Wah']
            expected = [h5_dset_2, h5_grp.ref]
            hdf_utils.check_and_link_ancillary(h5_dset_source, att_names,
                                               h5_main=None, anc_refs=expected)
            expected = [h5_dset_2, h5_grp]
            self.assertEqual(len(h5_dset_source.attrs.keys()), len(att_names))
            self.assertEqual(set(att_names), set(h5_dset_source.attrs.keys()))
            for name, exp_val in zip(att_names, expected):
                actual = h5_dset_source.attrs[name]
                self.assertIsInstance(actual, h5py.Reference)
                self.assertEqual(h5_f[actual], exp_val)
        os.remove(file_path)

    def test_too_many_names(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_2 = h5_f.create_dataset('Other', data=[1, 2, 3])
            _ = h5_f.create_group('Blah')
            att_names = ['Meh', 'Wah']
            expected = [h5_dset_2]
            hdf_utils.check_and_link_ancillary(h5_dset_source, att_names,
                                               h5_main=None, anc_refs=expected)
            att_names = [att_names[0]]
            print(list(h5_dset_source.attrs.keys()), att_names)
            self.assertEqual(len(h5_dset_source.attrs.keys()), len(att_names))
            self.assertEqual(set(att_names), set(h5_dset_source.attrs.keys()))
            for name, exp_val in zip(att_names, expected):
                actual = h5_dset_source.attrs[name]
                self.assertIsInstance(actual, h5py.Reference)
                self.assertEqual(h5_f[actual], exp_val)
        os.remove(file_path)

    def test_too_many_ancs(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_2 = h5_f.create_dataset('Other', data=[1, 2, 3])
            h5_grp = h5_f.create_group('Blah')
            att_names = ['Meh']
            expected = [h5_dset_2, h5_grp]
            hdf_utils.check_and_link_ancillary(h5_dset_source, att_names,
                                               h5_main=None, anc_refs=expected)
            expected = [expected[0]]
            self.assertEqual(len(h5_dset_source.attrs.keys()), len(att_names))
            self.assertEqual(set(att_names), set(h5_dset_source.attrs.keys()))
            for name, exp_val in zip(att_names, expected):
                actual = h5_dset_source.attrs[name]
                self.assertIsInstance(actual, h5py.Reference)
                self.assertEqual(h5_f[actual], exp_val)
        os.remove(file_path)

    def test_no_anc_refs_provided(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            with self.assertRaises(ValueError):
                hdf_utils.check_and_link_ancillary(h5_dset_source, ['M', 'Wa'],
                                                   h5_main=None, anc_refs=None)
        os.remove(file_path)

    def test_linking_main_legit_anc_names(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        shutil.copy(data_utils.std_beps_path, file_path)
        with h5py.File(file_path, mode='r+') as h5_f:
            h5_grp = h5_f['Raw_Measurement']
            h5_dset_source = h5_grp['Ancillary']
            h5_main = h5_grp['source_main']
            att_names = ['Spectroscopic_Values', 'Position_Indices']
            expected = [h5_grp[name] for name in att_names]
            hdf_utils.check_and_link_ancillary(h5_dset_source, att_names,
                                               h5_main=h5_main, anc_refs=None)
            self.assertEqual(len(h5_dset_source.attrs.keys()), len(att_names))
            self.assertEqual(set(att_names), set(h5_dset_source.attrs.keys()))
            for name, exp_val in zip(att_names, expected):
                actual = h5_dset_source.attrs[name]
                self.assertIsInstance(actual, h5py.Reference)
                self.assertEqual(h5_f[actual], exp_val)
        os.remove(file_path)

    def test_h5_main_non_dset_anc_names(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        shutil.copy(data_utils.std_beps_path, file_path)
        with h5py.File(file_path, mode='r+') as h5_f:
            h5_grp = h5_f['Raw_Measurement']
            h5_dset_source = h5_grp['Ancillary']
            h5_main = h5_grp['source_main']
            att_names = ['Spectroscopic_Values', 'quantity']
            hdf_utils.check_and_link_ancillary(h5_dset_source, att_names,
                                               h5_main=h5_main, anc_refs=None)
            # Should NOT copy the quantity attribute since it is not a dset
            att_names = [att_names[0]]
            expected = [h5_grp['Spectroscopic_Values']]
            self.assertEqual(set(att_names),
                             set(h5_dset_source.attrs.keys()))
            for name, exp_val in zip(att_names, expected):
                actual = h5_dset_source.attrs[name]
                self.assertIsInstance(actual, h5py.Reference)
                self.assertEqual(h5_f[actual], exp_val)
        os.remove(file_path)


class TestValidateAncH5Dsets(TestSimple):

    def test_valid_shapes(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_grp = h5_f['Raw_Measurement']
            h5_main = h5_grp['source_main']

            h5_pos_inds = h5_grp['Position_Indices']
            h5_pos_vals = h5_grp['Position_Values']

            hdf_utils.validate_anc_h5_dsets(h5_pos_inds, h5_pos_vals,
                                            h5_main.shape,
                                            is_spectroscopic=False)

            hdf_utils.validate_anc_h5_dsets(h5_pos_inds, h5_pos_vals,
                                            list(h5_main.shape),
                                            is_spectroscopic=False)

            h5_spec_inds = h5_grp['Spectroscopic_Indices']
            h5_spec_vals = h5_grp['Spectroscopic_Values']


            hdf_utils.validate_anc_h5_dsets(h5_spec_inds, h5_spec_vals,
                                            h5_main.shape,
                                            is_spectroscopic=True)

            hdf_utils.validate_anc_h5_dsets(h5_spec_inds, h5_spec_vals,
                                            list(h5_main.shape),
                                            is_spectroscopic=True)

    def test_invalid_spec_pos_bool(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_grp = h5_f['Raw_Measurement']
            h5_main = h5_grp['source_main']

            h5_pos_inds = h5_grp['Position_Indices']
            h5_pos_vals = h5_grp['Position_Values']

            with self.assertRaises(ValueError):
                hdf_utils.validate_anc_h5_dsets(h5_pos_inds, h5_pos_vals,
                                                h5_main.shape,
                                                is_spectroscopic=True)

            h5_spec_inds = h5_grp['Spectroscopic_Indices']
            h5_spec_vals = h5_grp['Spectroscopic_Values']

            with self.assertRaises(ValueError):
                hdf_utils.validate_anc_h5_dsets(h5_spec_inds, h5_spec_vals,
                                                h5_main.shape,
                                                is_spectroscopic=False)

    def test_mismatched_anc_shapes(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_grp = h5_f['Raw_Measurement']
            h5_main = h5_grp['source_main']

            h5_pos_inds = h5_grp['Position_Indices']
            h5_pos_vals = h5_grp['Ancillary']

            with self.assertRaises(ValueError):
                hdf_utils.validate_anc_h5_dsets(h5_pos_inds, h5_pos_vals,
                                                h5_main.shape,
                                                is_spectroscopic=False)

    def test_invalid_dtypes(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_grp = h5_f['Raw_Measurement']
            h5_main = h5_grp['source_main']

            h5_pos_inds = h5_grp['Position_Indices']
            h5_pos_vals = h5_grp['Position_Values']

            with self.assertRaises(TypeError):
                hdf_utils.validate_anc_h5_dsets('h5_pos_inds', h5_pos_vals,
                                                h5_main.shape,
                                                is_spectroscopic=False)

            with self.assertRaises(TypeError):
                hdf_utils.validate_anc_h5_dsets(h5_pos_inds, h5_pos_vals,
                                                np.arange(3),
                                                is_spectroscopic=False)


class TestValidateDimsAgainstMAin(unittest.TestCase):

    def test_single_dim(self):
        func = hdf_utils.validate_dims_against_main
        func((1, 5), write_utils.Dimension('blah', 'meh', np.arange(5)),
             is_spectroscopic=True)
        func((5, 1), [write_utils.Dimension('blah', 'meh', np.arange(5))],
             is_spectroscopic=False)

    def test_multi_dims(self):
        func = hdf_utils.validate_dims_against_main
        func((1, 15), [write_utils.Dimension('a', 'b', 5),
                       write_utils.Dimension('a', 'b', 3)],
             is_spectroscopic=True)
        func((15, 5), [write_utils.Dimension('a', 'b', 5),
                       write_utils.Dimension('a', 'b', 3)],
             is_spectroscopic=False)

    def test_invalid_dims(self):
        func = hdf_utils.validate_dims_against_main
        with self.assertRaises(ValueError):
            func((1, 25), [write_utils.Dimension('a', 'b', 5),
                           write_utils.Dimension('a', 'b', 3)],
                 is_spectroscopic=True)
        with self.assertRaises(ValueError):
            func((25, 5), [write_utils.Dimension('a', 'b', 5)],
                 is_spectroscopic=False)

    def test_invalid_dtypes(self):
        func = hdf_utils.validate_dims_against_main
        with self.assertRaises(TypeError):
            func('hello', write_utils.Dimension('a', 'b', 5))
        with self.assertRaises(TypeError):
            func((25, 5), "Dimension")

    def test_invalid_main_shape(self):
        func = hdf_utils.validate_dims_against_main
        with self.assertRaises(ValueError):
            func([-5, 1], write_utils.Dimension('a', 'b', 5))
        with self.assertRaises(ValueError):
            func([5], write_utils.Dimension('a', 'b', 5))
        with self.assertRaises(ValueError):
            func([1, 2, 5], write_utils.Dimension('a', 'b', 5))


class TestCopyLinkedObjects(TestSimple):

    def validate_copied_dataset(self, h5_f_new, h5_dest, dset_new_name,
                                dset_data, dset_attrs):
        self.assertTrue(dset_new_name in h5_f_new.keys())
        h5_anc_dest = h5_f_new[dset_new_name]
        self.assertIsInstance(h5_anc_dest, h5py.Dataset)
        self.assertTrue(np.allclose(dset_data, h5_anc_dest[()]))
        self.assertEqual(len(dset_attrs),
                         len(h5_anc_dest.attrs.keys()))
        for key, val in dset_attrs.items():
            self.assertEqual(val, h5_anc_dest.attrs[key])
        self.assertTrue(dset_new_name in h5_dest.attrs.keys())
        self.assertEqual(h5_f_new[h5_dest.attrs[dset_new_name]],
                         h5_anc_dest)

    def base_two_dsets_simple_attrs(self, exist_dset_same_data=False,
                                    exist_dset_diff_data_shape=False,
                                    exist_dset_diff_data=False,
                                    exist_grp_inst_dset=False):
        file_path = 'test.h5'
        new_path = 'new.h5'
        data_utils.delete_existing_file(file_path)
        data_utils.delete_existing_file(new_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_source = h5_f.create_dataset('Main', data=[1, 2, 3])
            simple_attrs = {'quantity': 'blah', 'units': 'nA'}
            h5_source.attrs.update(simple_attrs)

            h5_anc_1 = h5_f.create_dataset('Anc_1', data=[4, 5, 6])
            anc_1_attrs = {'a': 1, 'b': 3}
            h5_anc_1.attrs.update(anc_1_attrs)

            h5_anc_2 = h5_f.create_dataset('Anc_2', data=[7, 8, 9])
            anc_2_attrs = {'p': 78, 'j': 8}
            h5_anc_2.attrs.update(anc_2_attrs)

            h5_source.attrs['Pos_Inds'] = h5_anc_1.ref
            h5_source.attrs['Pos_Vals'] = h5_anc_2.ref

            with h5py.File(new_path, mode='w') as h5_f_new:
                h5_dest = h5_f_new.create_dataset('Duplicate', data=[1, 2, 3])

                if exist_dset_same_data:
                    _ = h5_f_new.create_dataset('Pos_Vals', data=[7, 8, 9])
                elif exist_dset_diff_data:
                    _ = h5_f_new.create_dataset('Pos_Vals', data=[8, 1, 3])
                elif exist_dset_diff_data_shape:
                    _ = h5_f_new.create_dataset('Pos_Vals',
                                                data=np.random.rand(5, 3))
                elif exist_grp_inst_dset:
                    _ = h5_f_new.create_group('Pos_Vals')

                if sys.version_info.major == 3 and exist_dset_same_data:
                    with self.assertWarns(UserWarning):
                        hdf_utils.copy_linked_objects(h5_source, h5_dest,
                                                      verbose=False)
                elif exist_dset_diff_data or exist_dset_diff_data_shape:
                    with self.assertRaises(ValueError):
                        hdf_utils.copy_linked_objects(h5_source, h5_dest,
                                                      verbose=False)
                elif exist_grp_inst_dset:
                    with self.assertRaises(TypeError):
                        hdf_utils.copy_linked_objects(h5_source, h5_dest,
                                                      verbose=False)
                else:
                    hdf_utils.copy_linked_objects(h5_source, h5_dest,
                                                  verbose=False)

                if not exist_dset_diff_data_shape and not exist_dset_diff_data\
                        and not exist_grp_inst_dset:
                    self.assertEqual(len(h5_f_new.keys()), 3)

                    self.validate_copied_dataset(h5_f_new, h5_dest, 'Pos_Inds',
                                                 h5_anc_1[()], anc_1_attrs)

                    self.validate_copied_dataset(h5_f_new, h5_dest, 'Pos_Vals',
                                                 h5_anc_2[()], anc_2_attrs)

        os.remove(file_path)
        os.remove(new_path)

    def test_two_dsets_simple_attrs_empty_dest(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=False,
                                         exist_dset_diff_data_shape=False,
                                         exist_dset_diff_data=False,
                                         exist_grp_inst_dset=False)

    def test_existing_anc_dset_same_data_no_attrs(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=True,
                                         exist_dset_diff_data_shape=False,
                                         exist_dset_diff_data=False,
                                         exist_grp_inst_dset=False)

    def test_existing_anc_dset_diff_data(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=False,
                                         exist_dset_diff_data_shape=False,
                                         exist_dset_diff_data=True,
                                         exist_grp_inst_dset=False)

    def test_existing_anc_dset_diff_data_shape(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=False,
                                         exist_dset_diff_data_shape=True,
                                         exist_dset_diff_data=False,
                                         exist_grp_inst_dset=False)

    def test_existing_group_instead_of_det(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=False,
                                         exist_dset_diff_data_shape=False,
                                         exist_dset_diff_data=False,
                                         exist_grp_inst_dset=True)


class TestCopyDataset(TestSimple):

    def validate_copied_dataset(self, h5_f_new, dset_new_name,
                                dset_data, dset_attrs):
        self.assertTrue(dset_new_name in h5_f_new.keys())
        h5_anc_dest = h5_f_new[dset_new_name]
        self.assertIsInstance(h5_anc_dest, h5py.Dataset)
        self.assertTrue(np.allclose(dset_data, h5_anc_dest[()]))
        self.assertEqual(len(dset_attrs),
                         len(h5_anc_dest.attrs.keys()))
        for key, val in dset_attrs.items():
            self.assertEqual(val, h5_anc_dest.attrs[key])

    def base_test(self, exist_dset_same_data=False, use_alias=False,
                  exist_dset_diff_data_shape=False, exist_dset_diff_data=False,
                  exist_grp_inst_dset=False):
        file_path = 'test.h5'
        new_path = 'new.h5'
        data_utils.delete_existing_file(file_path)
        data_utils.delete_existing_file(new_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_source = h5_f.create_dataset('Original', data=[1, 2, 3])
            simple_attrs = {'quantity': 'blah', 'units': 'nA'}
            h5_source.attrs.update(simple_attrs)

            with h5py.File(new_path, mode='w') as h5_f_new:

                if use_alias:
                    alias = 'Duplicate'
                else:
                    alias = 'Original'

                if exist_dset_same_data:
                    _ = h5_f_new.create_dataset(alias, data=[1, 2, 3])
                elif exist_dset_diff_data:
                    _ = h5_f_new.create_dataset(alias, data=[8, 1, 3])
                elif exist_dset_diff_data_shape:
                    _ = h5_f_new.create_dataset(alias,
                                                data=np.random.rand(5, 3))
                elif exist_grp_inst_dset:
                    _ = h5_f_new.create_group(alias)

                if use_alias:
                    al_arg = alias
                else:
                    al_arg = None

                func = hdf_utils.copy_dataset
                args = [h5_source, h5_f_new]
                kwargs = {'alias': al_arg, 'verbose': False}

                if exist_dset_diff_data or exist_dset_diff_data_shape:
                    with self.assertRaises(ValueError):
                        _ = func(*args, **kwargs)
                elif exist_grp_inst_dset:
                    with self.assertRaises(TypeError):
                        _ = func(*args, **kwargs)
                else:
                    _ = func(*args, **kwargs)

                if not exist_dset_diff_data_shape and not exist_dset_diff_data\
                        and not exist_grp_inst_dset:
                    self.assertEqual(len(h5_f_new.keys()), 1)
                    self.validate_copied_dataset(h5_f_new, alias,
                                                 h5_source[()], simple_attrs)

        os.remove(file_path)
        os.remove(new_path)

    def test_exact_copy(self):
        self.base_test(exist_dset_same_data=False, use_alias=False,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=False, exist_grp_inst_dset=False)

    def test_copy_w_alias(self):
        self.base_test(exist_dset_same_data=False, use_alias=True,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=False, exist_grp_inst_dset=False)

    def test_existing_group_same_name(self):
        self.base_test(exist_dset_same_data=False, use_alias=False,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=False, exist_grp_inst_dset=True)

    def test_existing_dset_same_name_data(self):
        self.base_test(exist_dset_same_data=True, use_alias=False,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=False, exist_grp_inst_dset=False)

    def test_existing_dset_same_name_diff_data_shape(self):
        self.base_test(exist_dset_same_data=False, use_alias=False,
                       exist_dset_diff_data_shape=True,
                       exist_dset_diff_data=False, exist_grp_inst_dset=False)

    def test_existing_dset_same_name_diff_data(self):
        self.base_test(exist_dset_same_data=False, use_alias=False,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=True, exist_grp_inst_dset=False)

"""
    def test_linking_main_plus_other_dsets(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        shutil.copy(data_utils.std_beps_path, file_path)
        with h5py.File(file_path, mode='r+') as h5_f:
            h5_grp = h5_f['Raw_Measurement']
            h5_dset_source = h5_grp['Ancillary']
            h5_main = h5_grp['source_main']
            att_names = ['Spectroscopic_Values', 'Position_Indices', 'X', 'Y']
            expected = [h5_grp[name] for name in att_names]
            hdf_utils.check_and_link_ancillary(h5_dset_source, att_names,
                                               h5_main=h5_main,
                                               anc_refs=expected[2:])
            for key, val in h5_dset_source.attrs.items():
                print(key, val)
            self.assertEqual(len(h5_dset_source.attrs.keys()), len(att_names))
            self.assertEqual(set(att_names), set(h5_dset_source.attrs.keys()))
            for name, exp_val in zip(att_names, expected):
                actual = h5_dset_source.attrs[name]
                self.assertIsInstance(actual, h5py.Reference)
                self.assertEqual(h5_f[actual], exp_val)
        os.remove(file_path)
"""


if __name__ == '__main__':
    unittest.main()
