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
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_dest = h5_f.create_dataset('Target', data=data)
            source_ref = h5_dset_source.regionref[0:-1:2]
            h5_dset_source.attrs['regref'] = source_ref

            hdf_utils.copy_region_refs(h5_dset_source, h5_dset_dest)

            self.assertTrue(np.allclose(h5_dset_source[h5_dset_source.attrs['regref']],
                                        h5_dset_dest[h5_dset_dest.attrs['regref']]))

        os.remove(file_path)

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

    def test_get_all_main_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/results_main']]
            main_dsets = hdf_utils.get_all_main(h5_f, verbose=False)
            # self.assertEqual(set(main_dsets), set(expected_dsets))
            self.assertEqual(len(main_dsets), len(expected_dsets))
            self.assertTrue(np.all([x.name == y.name for x, y in zip(main_dsets, expected_dsets)]))


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

            data_utils.validate_aux_dset_pair(self,h5_f, h5_inds, h5_vals, dim_names, dim_units, pos_data,
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

            data_utils.validate_aux_dset_pair(self, h5_group, h5_inds, h5_vals, dim_names, dim_units, spec_data,
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
            data_utils.validate_aux_dset_pair(self, h5_group, h5_inds, h5_vals, dim_names, dim_units, spec_inds,
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
