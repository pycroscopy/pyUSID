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
from ..io import data_utils
from .proc_utils import sho_slow_guess
from ..io.data_utils import *
from shutil import copyfile
#from pycroscopy.processing.signal_filter import SignalFilter
import tempfile
sys.path.append("../../../pyUSID/")
import pyUSID as usid


def _create_results_grp_dsets(h5_main, process_name, parms_dict):
    h5_results_grp = usid.hdf_utils.create_results_group(h5_main,
                                                         process_name)

    usid.hdf_utils.write_simple_attrs(h5_results_grp, parms_dict)

    spec_dims = usid.write_utils.Dimension('Empty', 'a. u.', 1)

    # 3. Create an empty results dataset that will hold all the results
    h5_results = usid.hdf_utils.write_main_dataset(
        h5_results_grp, (h5_main.shape[0], 1), 'Results',
        'quantity', 'units', None, spec_dims,
        dtype=np.float32,
        h5_pos_inds=h5_main.h5_pos_inds,
        h5_pos_vals=h5_main.h5_pos_vals)

    return h5_results_grp, h5_results


class NoMapFunc(usid.Process):

    def __init__(self, h5_main):
        parms_dict = {'parm_1': 1, 'parm_2': [1, 2, 3]}
        super(NoMapFunc, self).__init__(h5_main, 'Mean_Val',
                                                parms_dict=parms_dict)

    def _create_results_datasets(self):
        self.h5_results_grp, self.h5_results = _create_results_grp_dsets(self.h5_main,
                                                                         self.process_name,
                                                                         self.parms_dict)


class AvgSpecUltraBasic(usid.Process):

    def __init__(self, h5_main, *args, **kwargs):
        parms_dict = {'parm_1': 1, 'parm_2': [1, 2, 3]}
        super(AvgSpecUltraBasic, self).__init__(h5_main, 'Mean_Val',
                                                parms_dict=parms_dict,
                                                *args, **kwargs)

    def _create_results_datasets(self):
        self.h5_results_grp, self.h5_results = _create_results_grp_dsets(self.h5_main, self.process_name, self.parms_dict)

    @staticmethod
    def _map_function(spectrogram, *args, **kwargs):
        return np.mean(spectrogram)

    def _write_results_chunk(self):
        """
        Write the computed results back to the H5
        In this case, there isn't any more additional post-processing required
        """
        # Find out the positions to write to:
        pos_in_batch = self._get_pixels_in_current_batch()

        # write the results to the file
        self.h5_results[pos_in_batch, 0] = np.array(self._results)


class AvgSpecUltraBasicWTest(AvgSpecUltraBasic):

    def test(self, pos_ind):
        return np.mean(self.h5_main[pos_ind])


class AvgSpecUltraBasicWGetPrevResults(AvgSpecUltraBasic):

    def _get_existing_datasets(self):
        self.h5_results = self.h5_results_grp['Results']


class TestInvalidInitialization(unittest.TestCase):

    def test_no_map_func(self):
        delete_existing_file(data_utils.std_beps_path)
        data_utils.make_beps_file()
        self.h5_file = h5py.File(data_utils.std_beps_path, mode='r+')
        self.h5_main = self.h5_file['Raw_Measurement/source_main']
        self.h5_main = usid.USIDataset(self.h5_main)

        proc = NoMapFunc(self.h5_main)

        with self.assertRaises(NotImplementedError):
            _ = proc.compute()

        delete_existing_file(data_utils.std_beps_path)


    def test_read_only_file(self):
        delete_existing_file(data_utils.std_beps_path)
        data_utils.make_beps_file()
        self.h5_file = h5py.File(data_utils.std_beps_path, mode='r')
        self.h5_main = self.h5_file['Raw_Measurement/source_main']
        self.h5_main = usid.USIDataset(self.h5_main)

        with self.assertRaises(TypeError):
            _ = AvgSpecUltraBasic(self.h5_main)
        delete_existing_file(data_utils.std_beps_path)

    def test_not_main_dataset(self):
        delete_existing_file(data_utils.std_beps_path)
        data_utils.make_beps_file()
        self.h5_file = h5py.File(data_utils.std_beps_path, mode='r+')
        self.h5_main = self.h5_file['Raw_Measurement/X']

        with self.assertRaises(ValueError):
            _ = AvgSpecUltraBasic(self.h5_main)
        delete_existing_file(data_utils.std_beps_path)

    def test_invalid_process_name(self):
        delete_existing_file(data_utils.std_beps_path)
        data_utils.make_beps_file()
        self.h5_file = h5py.File(data_utils.std_beps_path, mode='r+')
        self.h5_main = self.h5_file['Raw_Measurement/source_main']
        self.h5_main = usid.USIDataset(self.h5_main)

        class TempProc(usid.Process):

            def __init__(self, h5_main, *args, **kwargs):
                parms_dict = {'parm_1': 1, 'parm_2': [1, 2, 3]}
                super(TempProc, self).__init__(h5_main, {'a': 1},
                                               parms_dict=parms_dict,
                                               *args, **kwargs)

        with self.assertRaises(TypeError):
            _ = TempProc(self.h5_main)
        delete_existing_file(data_utils.std_beps_path)

    def test_invalid_parms_dict(self):
        delete_existing_file(data_utils.std_beps_path)
        data_utils.make_beps_file()
        self.h5_file = h5py.File(data_utils.std_beps_path, mode='r+')
        self.h5_main = self.h5_file['Raw_Measurement/source_main']
        self.h5_main = usid.USIDataset(self.h5_main)

        class TempProc(usid.Process):

            def __init__(self, h5_main, *args, **kwargs):
                super(TempProc, self).__init__(h5_main, 'Proc',
                                               parms_dict='Parms',
                                               *args, **kwargs)

        with self.assertRaises(TypeError):
            _ = TempProc(self.h5_main)
        delete_existing_file(data_utils.std_beps_path)

    def test_none_parms_dict(self):
        delete_existing_file(data_utils.std_beps_path)
        data_utils.make_beps_file()
        self.h5_file = h5py.File(data_utils.std_beps_path, mode='r+')
        self.h5_main = self.h5_file['Raw_Measurement/source_main']
        self.h5_main = usid.USIDataset(self.h5_main)

        class TempProc(usid.Process):

            def __init__(self, h5_main, *args, **kwargs):
                super(TempProc, self).__init__(h5_main, 'Proc',
                                               parms_dict=None,
                                               *args, **kwargs)

        proc = TempProc(self.h5_main)
        self.assertEqual(proc.parms_dict, dict())
        delete_existing_file(data_utils.std_beps_path)


class TestCoreProcessNoTest(unittest.TestCase):

    def setUp(self, proc_class=AvgSpecUltraBasic):
        delete_existing_file(data_utils.std_beps_path)
        data_utils.make_beps_file()
        self.h5_file = h5py.File(data_utils.std_beps_path, mode='r+')
        self.h5_main = self.h5_file['Raw_Measurement/source_main']
        self.h5_main = usid.USIDataset(self.h5_main)
        self.exp_result = np.expand_dims(np.mean(self.h5_main[()], axis=1),
                                         axis=1)

        self.proc = proc_class(self.h5_main)

    def test_tfunc(self):
        with self.assertRaises(NotImplementedError):
            self.proc.test()

    def tearDown(self):
        delete_existing_file(data_utils.std_beps_path)

    def test_compute(self):
        h5_grp = self.proc.compute()
        self.assertIsInstance(h5_grp, h5py.Group)
        self.assertEqual(h5_grp, self.proc.h5_results_grp)
        results_dset = h5_grp['Results']
        self.assertTrue(np.allclose(results_dset[()], self.exp_result))
        # Verify status dataset has been written:
        self.assertTrue('completed_positions' in list(h5_grp.keys()))
        h5_status_dset = h5_grp['completed_positions']
        self.assertIsInstance(h5_status_dset, h5py.Dataset)
        self.assertEqual(h5_status_dset.shape, (self.h5_main.shape[0],))
        self.assertEqual(h5_status_dset.dtype, np.uint8)


class TestCoreProcessWTest(TestCoreProcessNoTest):

    def setUp(self, proc_class=AvgSpecUltraBasicWTest):
        super(TestCoreProcessWTest,
              self).setUp(proc_class=AvgSpecUltraBasicWTest)

    def test_tfunc(self):
        pix_ind = 5
        actual = self.proc.test(pix_ind)
        expected = self.exp_result[pix_ind]
        self.assertTrue(np.allclose(actual, expected))


class TestCoreProcessWExistingResults(unittest.TestCase):

    def __create_fake_result(self, percent_complete=100, parms_dict=None,
                             status_dset=True, status_attr=False,
                             verbose=False):

        if parms_dict is None:
            parms_dict = {'parm_1': 1, 'parm_2': [1, 2, 3]}

        if verbose:
            print('Using parms_dict: {}'.format(parms_dict))

        results_grp, h5_results_dset = _create_results_grp_dsets(
            self.h5_main, 'Mean_Val', parms_dict)

        # Intentionally set different results
        exp_result = np.expand_dims(np.random.rand(h5_results_dset.shape[0]),
                                    axis=1)
        h5_results_dset[:, 0] = exp_result[:, 0]

        # Build status:
        status = np.ones(shape=self.h5_main.shape[0], dtype=np.uint8)

        # Reset last portion of results to mean (expected)
        complete_index = int(self.h5_main.shape[0] * percent_complete / 100)
        if verbose:
            print('Positions up to {} of {} will be marked as complete'
                  '.'.format(complete_index, self.h5_main.shape[0]))

        if percent_complete < 100:
            # print('Reset results from position: {}'.format(complete_index))
            status[complete_index:] = 0
            # print(status)
            exp_result[complete_index:, 0] = np.mean(self.h5_main[complete_index:], axis=1)

        # 4. Create fake status dataset
        if status_dset:
            if verbose:
                print('Creating status dataset')
            _ = results_grp.create_dataset('completed_positions', data=status)
        if status_attr:
            if verbose:
                print('Writing legacy status attribute')
            results_grp.attrs['last_pixel'] = complete_index

        return results_grp, h5_results_dset, exp_result

    def setUp(self, proc_class=AvgSpecUltraBasicWGetPrevResults,
              percent_complete=100, parms_dict=None, status_dset=True,
              status_attr=False, verbose=False):
        delete_existing_file(data_utils.std_beps_path)
        data_utils.make_beps_file()
        self.h5_file = h5py.File(data_utils.std_beps_path, mode='r+')
        self.h5_main = self.h5_file['Raw_Measurement/source_main']
        self.h5_main = usid.USIDataset(self.h5_main)

        # Make some fake results here:
        if any([isinstance(item, (list, tuple)) for item in [percent_complete,
                                                             status_attr,
                                                             status_dset,
                                                             parms_dict]]):
            self.fake_results_grp = []
            self.h5_results = []
            self.exp_result = []

            for this_per, this_parms, has_status_dset, has_status_attr in zip(
                percent_complete, parms_dict, status_dset, status_attr):

                ret_vals = self.__create_fake_result(percent_complete=this_per,
                                                     parms_dict=this_parms,
                                                     status_dset=has_status_dset,
                                                     status_attr=has_status_attr,
                                                     verbose=verbose)
                self.fake_results_grp.append(ret_vals[0])
                self.h5_results.append(ret_vals[1])
                self.exp_result.append(ret_vals[2])
        else:
            ret_vals = self.__create_fake_result(percent_complete=percent_complete,
                                                 parms_dict=parms_dict,
                                                 status_dset=status_dset,
                                                 status_attr=status_attr,
                                                 verbose=verbose)
            self.fake_results_grp, self.h5_results, self.exp_result = ret_vals

        self.proc = AvgSpecUltraBasicWGetPrevResults(self.h5_main,
                                                     verbose=verbose)

    def test_compute(self):
        h5_results_grp = self.proc.compute(override=False)
        self.assertEqual(self.fake_results_grp, h5_results_grp)

    def tearDown(self):
        delete_existing_file(data_utils.std_beps_path)


class TestProcWLegacyResultsComplete(TestCoreProcessWExistingResults):

    def setUp(self, percent_complete=100):
        super(TestProcWLegacyResultsComplete,
              self).setUp(percent_complete=percent_complete,
                          status_dset=False, status_attr=True)

    def test_compute(self):
        super(TestProcWLegacyResultsComplete,
              self).test_compute()
        items_in_grp = list(self.proc.h5_results_grp.keys())
        # Should also have status dataset
        self.assertEqual(len(items_in_grp), 4)
        self.assertTrue(self.proc._status_dset_name in items_in_grp)
        status_dset = self.proc.h5_results_grp[self.proc._status_dset_name]
        self.assertEqual(np.sum(status_dset[()]), self.h5_main.shape[0])


class TestCoreProcessWDuplicateResultsOverride(TestCoreProcessWExistingResults):

    def test_compute(self, override=True):
        h5_grp = self.proc.compute(override=override)
        self.assertNotEqual(self.fake_results_grp, h5_grp)

        self.assertIsInstance(h5_grp, h5py.Group)
        self.assertTrue(h5_grp.name.endswith('001'))

        self.assertEqual(h5_grp, self.proc.h5_results_grp)
        results_dset = h5_grp['Results']
        self.assertTrue(np.allclose(results_dset[()],
                                    np.expand_dims(
                                        np.mean(self.h5_main[()], axis=1),
                                        axis=1)
                                    ))
        # Verify status dataset has been written:
        self.assertTrue('completed_positions' in list(h5_grp.keys()))
        h5_status_dset = h5_grp['completed_positions']
        self.assertIsInstance(h5_status_dset, h5py.Dataset)
        self.assertEqual(h5_status_dset.shape, (self.h5_main.shape[0],))
        self.assertEqual(h5_status_dset.dtype, np.uint8)


class TestCoreProcessWExistResultsDiffParms(TestCoreProcessWDuplicateResultsOverride):

    def setUp(self, proc_class=AvgSpecUltraBasicWGetPrevResults,
              percent_complete=100,
              parms_dict={'parm_1': 'Decoy', 'parm_2': 14.56}):
        super(TestCoreProcessWExistResultsDiffParms,
              self).setUp(percent_complete=percent_complete,
                          parms_dict=parms_dict)

    def test_duplicate_partial_results(self):
        self.assertEqual(len(self.proc.duplicate_h5_groups), 0)
        self.assertEqual(len(self.proc.partial_h5_groups), 0)

    def test_compute(self, override=False):
        super(TestCoreProcessWExistResultsDiffParms,
              self).test_compute(override=override)


class TestCoreProcessWExistingPartResults(TestCoreProcessWExistingResults):

    def setUp(self, proc_class=AvgSpecUltraBasicWGetPrevResults,
              percent_complete=50):
        super(TestCoreProcessWExistingPartResults,
              self).setUp(percent_complete=percent_complete)


class TestProcWLegacyResultsPartial(TestCoreProcessWExistingResults):

    def setUp(self, percent_complete=50):
        super(TestProcWLegacyResultsPartial,
              self).setUp(percent_complete=percent_complete,
                          status_dset=False, status_attr=True)

    def test_compute(self):
        super(TestProcWLegacyResultsPartial,
              self).test_compute()
        items_in_grp = list(self.proc.h5_results_grp.keys())
        self.assertEqual(len(items_in_grp), 4)
        self.assertTrue(self.proc._status_dset_name in items_in_grp)


class TestProcWoStatus(TestCoreProcessWDuplicateResultsOverride):

    def setUp(self, percent_complete=50):
        super(TestProcWoStatus,
              self).setUp(percent_complete=percent_complete,
                          status_dset=False, status_attr=False)

    def test_compute(self, override=False):
        super(TestProcWoStatus, self).test_compute(override=override)


class TestProcReturnCompletedNotPartial(TestCoreProcessWExistingResults):

    def setUp(self):
        super().setUp(percent_complete=[100, 50],
                      parms_dict=[None, None],
                      status_dset=[True, False],
                      status_attr=[False, True],
                      verbose=False)

    def test_compute(self):
        self.assertEqual(len(self.proc.duplicate_h5_groups), 1)
        self.assertEqual(self.proc.duplicate_h5_groups[0],
                         self.fake_results_grp[0])
        self.assertEqual(len(self.proc.partial_h5_groups), 1)
        self.assertEqual(self.proc.partial_h5_groups[0],
                         self.fake_results_grp[1])

        h5_results_grp = self.proc.compute(override=False)
        self.assertEqual(self.fake_results_grp[0], h5_results_grp)


class TestProcLastPartialResult(TestCoreProcessWExistingResults):

    def setUp(self):
        super().setUp(percent_complete=[75, 50],
                      parms_dict=[None, None],
                      status_dset=[True, False],
                      status_attr=[False, True],
                      verbose=False)

    def test_compute(self):
        self.assertEqual(len(self.proc.duplicate_h5_groups), 0)
        self.assertEqual(len(self.proc.partial_h5_groups), 2)
        for exp in self.fake_results_grp:
            self.assertTrue(exp in self.proc.partial_h5_groups)

        h5_results_grp = self.proc.compute(override=False)
        self.assertEqual(self.fake_results_grp[-1], h5_results_grp)


class TestUsePartialComputationIllegal(TestProcLastPartialResult):

    def test_compute(self):
        h5_grp = self.h5_main.parent.create_group('Blah')
        with self.assertRaises(ValueError):
            self.proc.use_partial_computation(h5_grp)


class TestUsePartialComputationLegit(TestProcLastPartialResult):

    def test_compute(self):
        self.proc.use_partial_computation(self.fake_results_grp[0])
        self.assertEqual(len(self.proc.duplicate_h5_groups), 0)
        self.assertEqual(len(self.proc.partial_h5_groups), 2)
        for exp in self.fake_results_grp:
            self.assertTrue(exp in self.proc.partial_h5_groups)

        h5_results_grp = self.proc.compute(override=False)
        self.assertEqual(self.fake_results_grp[0], h5_results_grp)


class TestMultiBatchCompute(TestCoreProcessNoTest):

    def test_compute(self):
        self.proc.verbose = True
        self.proc._max_pos_per_read = 6
        super().test_compute()

# TODO: interrupt computation
# TODO: set_cores
# TODO: set_memory
# TODO: Non-callable map_function or unit_computation
# TODO: lazy data chunk read
# TODO: parallel computation indeed being used (roughly N times faster)
# TODO: Timing per batch



orig_file_path = 'data/BELine_0004.h5'
temp_file_path = './BELine_0004.h5'


class TestProcess(unittest.TestCase):

    def setUp(self):
        delete_existing_file(temp_file_path)
        shutil.copy(orig_file_path, temp_file_path)

    def tearDown(self):
        delete_existing_file(temp_file_path)

'''

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
