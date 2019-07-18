"""
This script creates a partial h5py file then tests the process class with it.
Created on: Jul 12, 2019
Author: Emily Costa
"""

from data_utils import make_sparse_sampling_file
import pyUSID as usid
from pyUSID.io import dtype_utils, hdf_utils
import h5py
from pyUSID.processing.process import Process
import numpy as np
import os
import matplotlib.pyplot as plt

# Creates incomplete h5py dataset object in current path
h5_path = 'sparse_sampling.h5'
if not os.path.exists(h5_path):
    make_sparse_sampling_file()
h5_f = h5py.File(h5_path, mode='r+')
hdf_utils.print_tree(h5_f)
h5_main0 = h5_f['Measurement_000/Channel_000/Raw_Data']
h5_main1 = h5_f['Measurement_000/Channel_001/Raw_Data']

print(hdf_utils.simple.check_if_main(h5_main0, verbose=True))
#dtype_utils.check_dtype(h5_maini)

class SimpleProcess(Process):
    def __init__(self, h5_main, **kwargs):
        super(SimpleProcess, self).__init__(h5_main, **kwargs)
        self.data = None
        self.test_data = None
        self.results = None
        self.process_name = 'Simple_Process'

    def test(self):
        if self.mpi_rank > 0:
            return
        ran_ind = np.random.randint(0, high=self.h5_main.shape[0])
        self.test_data = np.fft.fftshift(np.fft.fft(self.h5_main[ran_ind]))
        
    def _create_results_datasets(self):
        self.h5_results_grp = hdf_utils.create_results_group(self.h5_main, self.process_name)
        assert isinstance(self.h5_results_grp, h5py.Group)
        self.results = hdf_utils.write_main_dataset(self.h5_results_grp,...)
        #...finish later
    def _write_results_chunk(self):
        self.results = self.h5_results_grp['Simple_Data']
    
    def _unit_computation(self):
        self.data = np.fft.fftshift(np.fft.fft(self.data, axis=1), axes=1)
    
    def plot_test(self):
        fig, axis = plt.subplots()
        axis.plot(self.test_data)
        plt.savefig('test_partial.png')

if __name__ == '__main__':
    simp = SimpleProcess(h5_main0)
    print(simp.test())
    simp.test()
    simp.plot_test()

