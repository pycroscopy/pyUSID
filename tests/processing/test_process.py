# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import shutil
import numpy as np
import h5py
import sys

sys.path.append("../../pyUSID/")
import pyUSID as usid

sys.path.append("./tests/io/")
import data_utils
from proc_utils import sho_slow_guess

orig_file_path = './data/BELine_0004.h5'
temp_file_path = './BELine_0004.h5'


class SuperBasicProcess(usid.Process):

    def _create_results_datasets(self):
        self.process_name = 'Peak_Finding'

        self.h5_results_grp = usid.hdf_utils.create_results_group(self.h5_main, self.process_name)

        usid.hdf_utils.write_simple_attrs(self.h5_results_grp,
                                          {'last_pixel': 0, 'algorithm': 'find_all_peaks'})

        results_shape = (self.h5_main.shape[0], 1)
        results_dset_name = 'Peak_Response'
        results_quantity = 'Amplitude'
        results_units = 'V'
        pos_dims = None  # Reusing those linked to self.h5_main
        spec_dims = usid.write_utils.Dimension('Empty', 'a. u.', 1)

        self.h5_results = usid.hdf_utils.write_main_dataset(self.h5_results_grp, results_shape,
                                                            results_dset_name,
                                                            results_quantity, results_units, pos_dims,
                                                            spec_dims,
                                                            dtype=np.float32,
                                                            h5_pos_inds=self.h5_main.h5_pos_inds,
                                                            h5_pos_vals=self.h5_main.h5_pos_vals)

    def _write_results_chunk(self):
        pos_in_batch = self._get_pixels_in_current_batch()
        self.h5_results[pos_in_batch, 0] = np.array(self._results)

    @staticmethod
    def _map_function(spectra, *args, **kwargs):

        sho_parms = sho_slow_guess(spectra, np.arange(spectra.size))

        return sho_parms[0]  # just the amplitude


class TestProcess(unittest.TestCase):

    def setUp(self):
        data_utils.delete_existing_file(temp_file_path)
        shutil.copy(orig_file_path, temp_file_path)

    def tearDown(self):
        data_utils.delete_existing_file(temp_file_path)

    def test_init_not_hdf5_dataset(self):
        pass

    def test_init_file_not_r_plus(self):
        pass

    def test_init_not_main_dset(self):
        pass


if __name__ == '__main__':
    unittest.main()