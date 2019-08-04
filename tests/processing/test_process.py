# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Emily Costa, Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import shutil
import numpy as np
import h5py
import sys
#from pycroscopy.processing.fft import LowPassFilter
from proc_utils import sho_slow_guess
import data_utils as data_utils
from shutil import copyfile
#from pycroscopy.processing.signal_filter import SignalFilter
import tempfile
from func_timeout import func_timeout, FunctionTimedOut

sys.path.append("../../../pyUSID/")
import pyUSID as usid

orig_file_path = 'data/BELine_0004.h5'
temp_file_path = './BELine_0004.h5'


class SuperBasicProcess(usid.Process):

    def __init__(self, h5_main):
        super(SuperBasicProcess, self).__init__(h5_main)

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

        file = np.arange(100).reshape(10,10)

        self.assertFalse(usid.hdf_utils.check_if_main(file))
'''
    def test_init_file_not_r_plus(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(temp_file_path, h5_path)

        h5_main = h5py.File(h5_path, mode='r')

        with self.assertRaises(TypeError):
            sbp = SuperBasicProcess(h5_main)

    def test_init_not_main_dset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(temp_file_path, h5_path)

        h5_main = h5py.File(h5_path, mode='r+')

        with self.assertRaises(ValueError):
            sbp = SuperBasicProcess(h5_main)

    def test_disruption(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        samp_rate = h5_grp.attrs['IO_samp_rate_[Hz]']
        num_spectral_pts = h5_main.shape[1]

        #frequency_filters = [fft.LowPassFilter(num_spectral_pts, samp_rate, 10E+3)]
        noise_tol = 1E-6

        sig_filt = SignalFilter(h5_main, noise_threshold=noise_tol, write_filtered=True,
                                write_condensed=False, num_pix=1, verbose=True)

        try:
            func_timeout(.1, sig_filt.compute, args=())
        except FunctionTimedOut:
            ("terminated compute")

    def test_try_resume_w_partial(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        process = usid.Process(h5_main, verbose=True)
        with self.assertRaises(ValueError):
            process.use_partial_computation(h5_partial_group=h5_main)

    def test_try_resume_wo_partial(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']
        # adds dummy data to duplicate group, simulates if there was duplicated
        process = usid.Process(h5_main, verbose=True)
        with self.assertRaises(ValueError):
            process.use_partial_computation(h5_partial_group=None)

    def test_found_duplicates(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']
        #adds dummy data to duplicate group, simulates if there was duplicated
        process = usid.Process(h5_main, verbose=True)
        process.process_name = 'process'
        process.duplicate_h5_groups = [h5_main]
        #just to finish test and make sure ran correctly
        groups = process._check_for_duplicates()
        boo = False
        if len(groups) > 0:
            boo = True

        self.assertTrue(boo)

    def test_found_partial(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']
        #adds dummy data to partial group, simulates if there was partial computation
        process = usid.Process(h5_main, verbose=True)
        process.process_name = 'process'
        process.partial_h5_groups = [h5_main]
        #just to finish test and make sure ran correctly
        groups = process._check_for_duplicates()
        boo = False
        if len(groups) > 0:
            boo = True

        self.assertTrue(boo)

    def test_test_raise(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        process = usid.Process(h5_main, verbose=True)

        with self.assertRaises(NotImplementedError):
            process.test()

    def test_map_raise(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        process = usid.Process(h5_main, verbose=True)

        with self.assertRaises(NotImplementedError):
            process._map_function()

    def test_write_results_chunk_raise(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        process = usid.Process(h5_main, verbose=True)

        with self.assertRaises(NotImplementedError):
            process._write_results_chunk()

    def test_create_results_raise(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        process = usid.Process(h5_main, verbose=True)

        with self.assertRaises(NotImplementedError):
            process._create_results_datasets()

    def test_parallel_compute(self):
        def inc(num):
            return num + 1
        dset = usid.parallel_compute(np.arange(100), inc, cores=3, verbose = True)


    def test_multi_node(self):
        pass

    def test_max_mem_mb(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        with self.assertRaises(TypeError):
            process = usid.Process(h5_main, cores=3, verbose=True, max_mem_mb=2500.5)
            process._set_memory_and_cores(cores=3, mem=None)

    def test_cores(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        process = usid.Process(h5_main, cores=3, verbose=True)

        with self.assertRaises(TypeError):
            process._set_memory_and_cores(cores=3.5, mem=None)

    def test_none_mpi_comm(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']


        process = usid.Process(h5_main, cores=3, verbose=True)
        process._set_memory_and_cores(cores=3, mem=None)

    def test_compute(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        samp_rate = h5_grp.attrs['IO_samp_rate_[Hz]']
        num_spectral_pts = h5_main.shape[1]

        frequency_filters = [LowPassFilter(num_spectral_pts, samp_rate, 10E+3)]
        noise_tol = 1E-6

        sig_filt = SignalFilter(h5_main, frequency_filters=frequency_filters,
                                noise_threshold=noise_tol, write_filtered=True,
                                write_condensed=False, num_pix=1,
                                verbose=True)

        h5_filt_grp = sig_filt.compute(override=True)

    def test_else_read_chunk(self):
        orig_path = 'data/pzt_nanocap_6_just_translation_copy.h5'

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = tmp_dir + 'gline.h5'
            copyfile(orig_path, h5_path)

        h5_f = h5py.File(h5_path, mode='r+')
        h5_grp = h5_f['Measurement_000/Channel_000']
        h5_main = h5_grp['Raw_Data']

        process = usid.Process(h5_main, cores=3, verbose=True)
        process._read_data_chunk()

        #self.assertEqual(process.data, None)
'''
if __name__ == '__main__':
    unittest.main()
