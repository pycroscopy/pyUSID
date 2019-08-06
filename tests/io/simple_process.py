"""
Simple process class for purpose of testing.
Created on: Jul 19, 2019
Author: Emily Costa
"""

import pyUSID as usid
from pyUSID.io import dtype_utils, hdf_utils
import h5py
from pyUSID.processing.process import Process
import numpy as np
import os
from pyUSID import hdf_utils
import matplotlib.pyplot as plt

class SimpleProcess(Process):
    def __init__(self, h5_main, verbose=True, **kwargs):
        super(SimpleProcess, self).__init__(h5_main, verbose, **kwargs)
        self.data = None
        self.test_data = None
        self.results = None
        self.chunk_amount = 0
        self.process_name = 'Simple_Process'
        if self.verbose: print('Done with initializing book-keepings')

    def test(self):
        if self.mpi_rank > 0:
            return
        ran_ind = np.random.randint(0, high=self.h5_main.shape[0])
        self.test_data = np.fft.fftshift(np.fft.fft(self.h5_main[ran_ind]))
        
    def _create_results_datasets(self):
        self.h5_results_grp = hdf_utils.create_results_group(self.h5_main, self.process_name)
        assert isinstance(self.h5_results_grp, h5py.Group)
        if self.verbose: print('Results group created.')
        self.results = hdf_utils.create_empty_dataset(self.h5_main, self.h5_main.dtype, 'Filtered_Data',
                                                    h5_group=self.h5_results_grp)
        #self.results = hdf_utils.write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], 1), "Results", "Results", "Units", None,
        #usid.io.write_utils.Dimension('arb', '', [1]), h5_pos_inds=self.h5_main.h5_pos_inds, h5_pos_vals=self.h5_main.h5_pos_vals, dtype=np.float32)
        if self.verbose: print('Empty main dataset for results written')

    def _write_results_chunk(self):
        pos_in_batch = self._get_pixels_in_current_batch()
        print(type(self.data))
        print(type(self.results))
        self.results[pos_in_batch, :] = self.data
        #self.results = self.h5_results_grp['Simple_Data']
        self.chunk_amount = self.chunk_amount + 1
        if self.verbose: print('Chunk {} written.'.format(self.chunk_amount))
    
    def _unit_computation(self):
        self.data = np.fft.fftshift(np.fft.fft(self.data, axis=1), axes=1)
    
    def plot_test(self):
        fig, axis = plt.subplots()
        axis.plot(self.test_data)
        plt.savefig('test_partial.png')
        if self.verbose: print('Test image created.')



