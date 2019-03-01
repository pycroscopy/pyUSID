"""
:class:`~pyUSID.processing.process.Process` - An abstract class for formulating scientific problems as computational
problems

Created on 7/17/16 10:08 AM

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import psutil
import time as tm
import h5py
from numbers import Number
from multiprocessing import cpu_count

from .comp_utils import parallel_compute, get_MPI, group_ranks_by_socket, get_available_memory
from ..io.hdf_utils import check_if_main, check_for_old, get_attributes
from ..io.usi_data import USIDataset
from ..io.dtype_utils import integers_to_slices
from ..io.io_utils import format_time, format_size


class Process(object):
    """
    An abstract class for formulating scientific problems as computational problems. This class handles the tedious,
    science-agnostic, file-operations, parallel-computations, and book-keeping operations such that children classes
    only need to specify application-relevant code for processing the data.
    """

    def __init__(self, h5_main, cores=None, max_mem_mb=4*1024, verbose=False):
        """
        Parameters
        ----------
        h5_main : :class:`~pyUSID.io.usi_data.USIDataset`
            The USID main HDF5 dataset over which the analysis will be performed.
        cores : uint, optional
            How many cores to use for the computation. Default: all available cores - 2 if operating outside MPI context
        max_mem_mb : uint, optional
            How much memory to use for the computation.  Default 1024 Mb
        verbose : bool, Optional, default = False
            Whether or not to print debugging statements
        """

        if h5_main.file.mode != 'r+':
            raise TypeError('Need to ensure that the file is in r+ mode to write results back to the file')

        MPI = get_MPI()

        if MPI is not None:
            # If we came here then, the user has intentionally asked for multi-node computation
            comm = MPI.COMM_WORLD
            self.mpi_comm = comm
            self.mpi_rank = comm.Get_rank()
            self.mpi_size = comm.Get_size()

            if verbose:
                print("Rank {} of {} on {} sees {} logical cores on the socket".format(comm.Get_rank(), comm.Get_size(),
                                                                                       MPI.Get_processor_name(),
                                                                                       cpu_count()))

            # First, ensure that cores=logical cores in node. No point being economical / considerate
            cores = psutil.cpu_count()

            # It is sufficient if just one rank checks all this.
            if self.mpi_rank == 0:
                print('Working on {} ranks via MPI'.format(self.mpi_size))

            # Ensure that the file is opened in the correct comm or something
            if h5_main.file.driver != 'mpio':
                raise TypeError('The HDF5 file should have been opened with driver="mpio". Current driver = "{}"'
                                ''.format(h5_main.file.driver))

            """
            # Not sure how to check for this correctly
            messg = None
            try:
                if h5_main.file.comm != comm:
                    messg = 'The HDF5 file should have been opened with comm=MPI.COMM_WORLD. Currently comm={}'
                            ''.format(h5_main.file.comm)
            except AttributeError:
                messg = 'The HDF5 file should have been opened with comm=MPI.COMM_WORLD'
            if messg is not None:
                raise TypeError(messg)
            """

        else:
            if verbose:
                print('No mpi4py found or script was not called via mpixexec / mpirun. '
                      'Assuming single node computation')
            self.mpi_comm = None
            self.mpi_size = 1
            self.mpi_rank = 0

        # Checking if dataset is "Main"
        if not check_if_main(h5_main, verbose=verbose and self.mpi_rank == 0):
            raise ValueError('Provided dataset is not a "Main" dataset with necessary ancillary datasets')

        if MPI is not None:
            MPI.COMM_WORLD.barrier()
        # Not sure if we need a barrier here.

        # Saving these as properties of the object:
        self.h5_main = USIDataset(h5_main)
        self.verbose = verbose
        self._cores = None
        self.__ranks_on_socket = 1
        self.__socket_master_rank = 0
        self._max_pos_per_read = None
        self._max_mem_mb = None

        # Now have to be careful here since the below properties are a function of the MPI rank
        self.__start_pos = None
        self.__rank_end_pos = None
        self.__end_pos = None
        self.__pixels_in_batch = None

        # Determining the max size of the data that can be put into memory
        # all ranks go through this and they need to have this value any
        self._set_memory_and_cores(cores=cores, mem=max_mem_mb)
        self.duplicate_h5_groups = []
        self.partial_h5_groups = []
        self.process_name = None  # Reset this in the extended classes
        self.parms_dict = None

        """
        The name of the HDF5 dataset that should be present to signify which positions have already been computed
        This is NOT a fully private variable so that multiple processes can be run within a single group - Eg Fitter
        In the case of Fitter - this name can be changed from 'completed_guesses' to 'completed_fits'
        check_for_duplicates will be called by the Child class where they have the opportunity to change this
        variable before checking for duplicates
        """
        self._status_dset_name = 'completed_positions'

        self._results = None
        self.h5_results_grp = None

        # Check to see if the resuming feature has been implemented:
        self.__resume_implemented = False
        try:
            self._get_existing_datasets()
        except NotImplementedError:
            if verbose and self.mpi_rank == 0:
                print('It appears that this class may not be able to resume computations')
        except:
            # NameError for variables that don't exist
            # AttributeError for self.var_name that don't exist
            # TypeError (NoneType) etc.
            self.__resume_implemented = True

        if self.mpi_rank == 0:
            print('Consider calling test() to check results before calling compute() which computes on the entire'
                  ' dataset and writes back to the HDF5 file')

        # DON'T check for duplicates since parms_dict has not yet been initialized.
        # Sub classes will check by themselves if they are interested.

    def __assign_job_indices(self):
        """
        Sets the start and end indices for each MPI rank
        """
        # First figure out what positions need to be computed
        self._compute_jobs = np.where(self._h5_status_dset[()] == 0)[0]
        if self.verbose and self.mpi_rank == 0:
            print('Among the {} positions in this dataset, the following positions need to be computed: {}'
                  '.'.format(self.h5_main.shape[0], self._compute_jobs))

        pos_per_rank = self._compute_jobs.size // self.mpi_size  # integer division
        if self.verbose and self.mpi_rank == 0:
            print('Each rank is required to work on {} of the {} (remaining) positions in this dataset'
                  '.'.format(pos_per_rank, self._compute_jobs.size))

        # The start and end indices now correspond to the indices in the incomplete jobs rather than the h5 dataset
        self.__start_pos = self.mpi_rank * pos_per_rank
        self.__rank_end_pos = (self.mpi_rank + 1) * pos_per_rank
        self.__end_pos = int(min(self.__rank_end_pos, self.__start_pos + self._max_pos_per_read))
        if self.mpi_rank == self.mpi_size - 1:
            # Force the last rank to go to the end of the dataset
            self.__rank_end_pos = self._compute_jobs.size

        if self.verbose:
            print('Rank {} will read positions {} to {} of {}'.format(self.mpi_rank, self.__start_pos,
                                                                      self.__rank_end_pos, self.h5_main.shape[0]))

    def _estimate_compute_time_per_pixel(self, *args, **kwargs):
        """
        Estimates how long it takes to compute an average pixel's worth of data. This information should be used by the
        user to limit the number of pixels that will be processed per batch to make best use of check-pointing. This
        function is exposed to the developer of the child classes. An approximate can be derived if it is simpler

        Returns
        -------

        """
        chosen_pos = np.random.randint(0, high=self.h5_main.shape[0]-1, size=5)
        t0 = tm.time()
        _ = parallel_compute(self.h5_main[chosen_pos, :], self._map_function, cores=1,
                             lengthy_computation=False, func_args=args, func_kwargs=kwargs, verbose=False)
        return (tm.time() - t0) / len(chosen_pos)

    def _get_pixels_in_current_batch(self):
        """
        Returns the indices of the pixels that will be processed in this batch.

        Returns
        -------
        pixels_in_batch : :class:`numpy.ndarray`
            1D array of unsigned integers denoting the pixels that will be read, processed, and written back to
        """
        return self.__pixels_in_batch

    def test(self, **kwargs):
        """
        Tests the process on a subset (for example a pixel) of the whole data. The class can be re-instantiated with
        improved parameters and tested repeatedly until the user is content, at which point the user can call
        :meth:`~pyUSID.processing.process.Process.compute` on the whole dataset.

        Notes
        -----
        This is not a function that is expected to be called in MPI

        Parameters
        ----------
        kwargs - dict, optional
            keyword arguments to test the process
        Returns
        -------
        """
        # All children classes should call super() OR ensure that they only work for self.mpi_rank == 0
        raise NotImplementedError('test_on_subset has not yet been implemented')

    def _check_for_duplicates(self):
        """
        Checks for instances where the process was applied to the same dataset with the same parameters

        Returns
        -------
        duplicate_h5_groups : list of h5py.Group objects
            List of groups satisfying the above conditions
        """
        if self.verbose and self.mpi_rank == 0:
            print('Checking for duplicates:')

        # This list will contain completed runs only
        duplicate_h5_groups = check_for_old(self.h5_main, self.process_name, new_parms=self.parms_dict)
        partial_h5_groups = []

        # First figure out which ones are partially completed:
        if len(duplicate_h5_groups) > 0:
            for index, curr_group in enumerate(duplicate_h5_groups):
                """
                Earlier, we only checked the 'last_pixel' but to be rigorous we should check self._status_dset_name
                The last_pixel attribute check may be deprecated in the future.
                Note that legacy computations did not have this dataset. We can add to partially computed datasets
                """
                if self._status_dset_name in curr_group.keys():

                    # Case 1: Modern Process results:
                    status_dset = curr_group[self._status_dset_name]

                    if not isinstance(status_dset, h5py.Dataset):
                        # We should not come here if things were implemented correctly
                        if self.mpi_rank == 0:
                            print('Results group: {} contained an object named: {} that should have been a dataset'
                                  '.'.format(curr_group, self._status_dset_name))

                    if self.h5_main.shape[0] != status_dset.shape[0] or len(status_dset.shape) > 1 or \
                            status_dset.dtype != np.uint8:
                        if self.mpi_rank == 0:
                            print('Status dataset: {} was not of the expected shape or datatype'.format(status_dset))

                    # Finally, check how far the computation was completed.
                    if len(np.where(status_dset[()] == 0)[0]) != 0:  # If there are pixels uncompleted
                        # remove from duplicates and move to partial
                        partial_h5_groups.append(duplicate_h5_groups.pop(index))
                        # Let's write the legacy attribute for safety
                        curr_group.attrs['last_pixel'] = self.h5_main.shape[0]
                        # No further checks necessary
                        continue
                    else:
                        # Optionally calculate how much was completed:
                        if self.mpi_rank == 0:
                            if len(np.where(status_dset[()] == 0)[0]) > 0:  # if there are unfinished pixels
                                percent_complete = int(100 * len(np.where(status_dset[()] == 0)[0]) / status_dset.shape[0])
                                print('Group: {}: computation was {}% completed'.format(curr_group, percent_complete))

                # Case 2: Legacy results group:
                if 'last_pixel' not in curr_group.attrs.keys():
                    if self.mpi_rank == 0:
                        # Should not be coming here at all
                        print('Group: {} had neither the status HDF5 dataset or the legacy attribute: "last_pixel"'
                              '.'.format(curr_group))
                    # Not sure what to do with such groups. Don't consider them in the future
                    duplicate_h5_groups.pop(index)
                    continue

                # Finally, do the legacy test:
                if curr_group.attrs['last_pixel'] < self.h5_main.shape[0]:
                    # Should we create the dataset here, to make the group future-proof?
                    # remove from duplicates and move to partial
                    partial_h5_groups.append(duplicate_h5_groups.pop(index))

        if len(duplicate_h5_groups) > 0 and self.mpi_rank == 0:
            print('Note: ' + self.process_name + ' has already been performed with the same parameters before. '
                                                 'These results will be returned by compute() by default. '
                                                 'Set override to True to force fresh computation')
            print(duplicate_h5_groups)

        if len(partial_h5_groups) > 0 and self.mpi_rank == 0:
            print('Note: ' + self.process_name + ' has already been performed PARTIALLY with the same parameters. '
                                                 'compute() will resuming computation in the last group below. '
                                                 'To choose a different group call use_patial_computation()'
                                                 'Set override to True to force fresh computation or resume from a '
                                                 'data group besides the last in the list.')
            print(partial_h5_groups)

        return duplicate_h5_groups, partial_h5_groups

    def use_partial_computation(self, h5_partial_group=None):
        """
        Extracts the necessary parameters from the provided h5 group to resume computation

        Parameters
        ----------
        h5_partial_group : :class:`h5py.Group`
            Group containing partially computed results
        """
        # Attempt to automatically take partial results
        if h5_partial_group is None:
            if len(self.partial_h5_groups) < 1:
                raise ValueError('No group was found with partial results and no such group was provided')
            h5_partial_group = self.partial_h5_groups[-1]
        else:
            # Make sure that this group is among the legal ones already discovered:
            if h5_partial_group not in self.partial_h5_groups:
                raise ValueError('Provided group does not appear to be in the list of discovered groups')

        self.parms_dict = get_attributes(h5_partial_group)

        self.h5_results_grp = h5_partial_group

    def _set_memory_and_cores(self, cores=None, mem=None):
        """
        Checks hardware limitations such as memory, number of CPU cores and sets the recommended data chunk sizes and
        the number of cores to be used by analysis methods. This function can work with clusters with heterogeneous
        memory sizes (e.g. CADES SHPC Condo).

        Parameters
        ----------
        cores : uint, optional, Default = 1
            How many cores to use for the computation.
        mem : uint, optional, Default = 1024
            The amount a memory in Mb to use in the computation
        """
        if self.mpi_comm is None:
            min_free_cores = 1 + int(psutil.cpu_count() > 4)

            if cores is None:
                self._cores = max(1, psutil.cpu_count() - min_free_cores)
            else:
                if not isinstance(cores, int):
                    raise TypeError('cores should be an integer but got: {}'.format(cores))
                cores = int(abs(cores))
                self._cores = max(1, min(psutil.cpu_count(), cores))

            self.__socket_master_rank = 0
            self.__ranks_on_socket = 1
        else:
            # user-provided input cores will simply be ignored in an effort to use the entire CPU
            ranks_by_socket = group_ranks_by_socket(verbose=False)
            self.__socket_master_rank = ranks_by_socket[self.mpi_rank]
            # which ranks in this socket?
            ranks_on_this_socket = np.where(ranks_by_socket == self.__socket_master_rank)[0]
            # how many in this socket?
            self.__ranks_on_socket = ranks_on_this_socket.size
            # Force usage of all available memory
            mem = None
            self._cores = 1
            # Disabling the following line since mpi4py and joblib didn't play well for Bayesian Inference
            # self._cores = self.__cores_per_rank = psutil.cpu_count() // self.__ranks_on_socket

        # TODO: Convert all to bytes!
        _max_mem_mb = get_available_memory() / 1024 ** 2  # in MB
        if mem is None:
            mem = _max_mem_mb
        else:
            if not isinstance(mem, int):
                raise TypeError('mem must be a whole number')
            mem = abs(mem)

        self._max_mem_mb = min(_max_mem_mb, mem)

        # Remember that multiple processes (either via MPI or joblib) will share this socket
        max_data_chunk = self._max_mem_mb / (self._cores * self.__ranks_on_socket)

        # Now calculate the number of positions OF RAW DATA ONLY that can be stored in memory in one go PER RANK
        mb_per_position = self.h5_main.dtype.itemsize * self.h5_main.shape[1] / 1024 ** 2
        self._max_pos_per_read = int(np.floor(max_data_chunk / mb_per_position))

        if self.verbose and self.mpi_rank == self.__socket_master_rank:
            # expected to be the same for all ranks so just use this.
            print('Rank {} - on socket with {} logical cores and {} avail. RAM shared by {} ranks each given {} cores'
                  '.'.format(self.__socket_master_rank, psutil.cpu_count(), format_size(_max_mem_mb * 1024**2, 2), 
                             self.__ranks_on_socket, self._cores))
            print('Allowed to read {} pixels per chunk'.format(self._max_pos_per_read))

    @staticmethod
    def _map_function(*args, **kwargs):
        """
        The function that manipulates the data on a single instance (position). This will be used by
        :meth:`~pyUSID.processing.process.Process._unit_computation` to process a chunk of data in parallel

        Parameters
        ----------
        args : list
            arguments to the function in the correct order
        kwargs : dict
            keyword arguments to the function
        Returns
        -------
        object
        """
        raise NotImplementedError('Please override the _unit_function specific to your process')

    def _read_data_chunk(self):
        """
        Reads a chunk of data for the intended computation into memory
        """
        if self.__start_pos < self.__rank_end_pos:
            self.__end_pos = int(min(self.__rank_end_pos, self.__start_pos + self._max_pos_per_read))

            # DON'T DIRECTLY apply the start and end indices anymore to the h5 dataset. Find out what it means first
            self.__pixels_in_batch = self._compute_jobs[self.__start_pos: self.__end_pos]
            self.data = self.h5_main[self.__pixels_in_batch, :]
            if self.verbose:
                print('Rank {} - Read positions: {}'.format(self.mpi_rank, self.__pixels_in_batch, self.__rank_end_pos))

            # DON'T update the start position

        else:
            if self.verbose:
                print('Rank {} - Finished reading all data!'.format(self.mpi_rank))
            self.data = None

    def _write_results_chunk(self):
        """
        Writes the computed results into appropriate datasets.
        This needs to be rewritten since the processed data is expected to be at least as large as the dataset
        """
        # Now update the start position
        self.__start_pos = self.__end_pos
        # This line can remain as is
        raise NotImplementedError('Please override the _set_results specific to your process')

    def _create_results_datasets(self):
        """
        Process specific call that will write the h5 group, guess dataset, corresponding spectroscopic datasets and also
        link the guess dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.
        """
        raise NotImplementedError('Please override the _create_results_datasets specific to your process')

    def __create_compute_status_dataset(self):
        """
        Creates a dataset that keeps track of what pixels / rows have already been computed. Users are not expected to
        extend / modify this function.
        """
        # Check to make sure that such a group doesn't already exist
        if self._status_dset_name in self.h5_results_grp.keys():
            self._h5_status_dset = self.h5_results_grp[self._status_dset_name]
            if not isinstance(self._h5_status_dset, h5py.Dataset):
                raise ValueError('Provided results group: {} contains an expected object ({}) that is not a dataset'
                                 '.'.format(self.h5_results_grp, self._h5_status_dset))
            if self.h5_main.shape[0] != self._h5_status_dset.shape[0] or len(self._h5_status_dset.shape) > 1 or \
                    self._h5_status_dset.dtype != np.uint8:
                if self.mpi_rank == 0:
                    raise ValueError('Status dataset: {} was not of the expected shape or datatype'
                                     '.'.format(self._h5_status_dset))
        else:
            self._h5_status_dset = self.h5_results_grp.create_dataset(self._status_dset_name, dtype=np.uint8,
                                                                      shape=(self.h5_main.shape[0],))
            #  Could be fresh computation or resuming from a legacy computation
            if 'last_pixel' in self.h5_results_grp.attrs.keys():
                completed_pixels = self.h5_results_grp.attrs['last_pixel']
                if completed_pixels > 0:
                    self._h5_status_dset[:completed_pixels] = 1

    def _get_existing_datasets(self):
        """
        The purpose of this function is to allow processes to resume from partly computed results
        Start with self.h5_results_grp
        """
        raise NotImplementedError('Please override the _get_existing_datasets specific to your process')

    def _unit_computation(self, *args, **kwargs):
        """
        The unit computation that is performed per data chunk. This allows room for any data pre / post-processing
        as well as multiple calls to parallel_compute if necessary
        """
        # TODO: Try to use the functools.partials to preconfigure the map function
        # cores = number of processes / rank here
        self._results = parallel_compute(self.data, self._map_function, cores=self._cores,
                                         lengthy_computation=False,
                                         func_args=args, func_kwargs=kwargs,
                                         verbose=self.verbose)

    def compute(self, override=False, *args, **kwargs):
        """
        Creates placeholders for the results, applies the :meth:`~pyUSID.processing.process.Process._unit_computation`
        to chunks of the dataset

        Parameters
        ----------
        override : bool, optional. default = False
            By default, compute will simply return duplicate results to avoid recomputing or resume computation on a
            group with partial results. Set to True to force fresh computation.
        args : list
            arguments to the mapped function in the correct order
        kwargs : dict
            keyword arguments to the mapped function

        Returns
        -------
        h5_results_grp : :class:`h5py.Group`
            Group containing all the results
        """

        class SimpleFIFO(object):
            """
            Simple class that maintains a moving average of some numbers.
            """

            def __init__(self, length=5):
                """
                Create a SimpleFIFO object

                Parameters
                ----------
                length : unsigned integer
                    Number of values that need to be maintained for the moving average
                """
                self.__queue = list()
                if not isinstance(length, int):
                    raise TypeError('length must be a positive integer')
                if length <= 0:
                    raise ValueError('length must be a positive integer')
                self.__max_length = length
                self.__count = 0

            def put(self, item):
                """
                Adds the item to the internal queue. If the size of the queue exceeds its capacity, the oldest
                item is removed.

                Parameters
                ----------
                item : float or int
                    Any real valued number
                """
                if (not isinstance(item, Number)) or isinstance(item, complex):
                    raise TypeError('Provided item: {} is not a Number'.format(item))
                self.__queue.append(item)
                self.__count += 1
                if len(self.__queue) > self.__max_length:
                    _ = self.__queue.pop(0)

            def get_mean(self):
                """
                Returns the average of the elements within the queue

                Returns
                -------
                avg : number.Number
                    Mean of all elements within the queue
                """
                return np.mean(self.__queue)

            def get_cycles(self):
                """
                Returns the number of items that have been added to the queue in total

                Returns
                -------
                count : int
                    number of items that have been added to the queue in total
                """
                return self.__count

        if not override:
            if len(self.duplicate_h5_groups) > 0:
                if self.mpi_rank == 0:
                    print('Returned previously computed results at ' + self.duplicate_h5_groups[-1].name)
                return self.duplicate_h5_groups[-1]
            elif len(self.partial_h5_groups) > 0:
                if self.mpi_rank == 0:
                    print('Resuming computation in group: ' + self.partial_h5_groups[-1].name)
                self.use_partial_computation()

        resuming = False
        if self.h5_results_grp is None:
            # starting fresh
            if self.verbose and self.mpi_rank == 0:
                print('Creating HDF5 group and datasets to hold results')
            self._create_results_datasets()
        else:
            # resuming from previous checkpoint
            resuming = True
            self._get_existing_datasets()

        self.__create_compute_status_dataset()

        if resuming and self.mpi_rank == 0:
            percent_complete = int(100 * len(np.where(self._h5_status_dset[()] == 0)[0]) /
                                   self._h5_status_dset.shape[0])
            print('Resuming computation. {}% completed already'.format(percent_complete))

        self.__assign_job_indices()

        # Not sure if this is necessary but I don't think it would hurt either
        if self.mpi_comm is not None:
            self.mpi_comm.barrier()

        compute_times = SimpleFIFO(5)
        write_times = SimpleFIFO(5)
        orig_rank_start = self.__start_pos

        if self.mpi_rank == 0 and self.mpi_size == 1:
            if self.__resume_implemented:
                print('\tThis class (likely) supports interruption and resuming of computations!\n'
                      '\tIf you are operating in a python console, press Ctrl+C or Cmd+C to abort\n'
                      '\tIf you are in a Jupyter notebook, click on "Kernel">>"Interrupt"\n'
                      '\tIf you are operating on a cluster and your job gets killed, re-run the job to resume\n')
            else:
                print('\tThis class does NOT support interruption and resuming of computations.\n'
                      '\tIn order to enable this feature, simply implement the _get_existing_datasets() function')

        if self.verbose and self.mpi_rank == self.__socket_master_rank:
            print('Rank: {} - with nothing loaded has {} free memory'
                  ''.format(self.mpi_rank, format_size(get_available_memory())))

        self._read_data_chunk()

        if self.mpi_comm is not None:
            self.mpi_comm.barrier()

        if self.verbose and self.mpi_rank == self.__socket_master_rank:
            print('Rank: {} - with only raw data loaded has {} free memory'
                  ''.format(self.mpi_rank, format_size(get_available_memory())))

        while self.data is not None:

            num_jobs_in_batch = self.__end_pos - self.__start_pos

            t_start_1 = tm.time()

            self._unit_computation(*args, **kwargs)

            comp_time = np.round(tm.time() - t_start_1, decimals=2)  # in seconds
            time_per_pix = comp_time / num_jobs_in_batch
            compute_times.put(time_per_pix)

            if self.verbose:
                print('Rank {} - computed chunk in {} or {} per pixel. Average: {} per pixel'
                      '.'.format(self.mpi_rank, format_time(comp_time), format_time(time_per_pix),
                                 format_time(compute_times.get_mean())))

            # Ranks can become memory starved. Check memory usage - raw data + results in memory at this point
            if self.verbose and self.mpi_rank == self.__socket_master_rank:
                print('Rank: {} - now holding onto raw data + results has {} free memory'
                      ''.format(self.mpi_rank, format_size(get_available_memory())))

            t_start_2 = tm.time()
            self._write_results_chunk()

            # NOW, update the positions. Users are NOT allowed to touch start and end pos
            self.__start_pos = self.__end_pos
            # Leaving in this provision that will allow restarting of processes
            if self.mpi_size == 1:
                self.h5_results_grp.attrs['last_pixel'] = self.__end_pos
            # Child classes don't even have to worry about flushing. Process will do it.
            self.h5_main.file.flush()

            dump_time = np.round(tm.time() - t_start_2, decimals=2)
            write_times.put(dump_time / num_jobs_in_batch)

            if self.verbose:
                print('Rank {} - wrote its {} pixel chunk in {}'.format(self.mpi_rank,
                                                                        num_jobs_in_batch,
                                                                        format_time(dump_time)))

            time_remaining = (self.__rank_end_pos - self.__end_pos) * \
                             (compute_times.get_mean() + write_times.get_mean())

            if self.verbose or self.mpi_rank == 0:
                percent_complete = int(100 * (self.__end_pos - orig_rank_start) /
                                       (self.__rank_end_pos - orig_rank_start))
                print('Rank {} - {}% complete. Time remaining: {}'.format(self.mpi_rank, percent_complete,
                                                                          format_time(time_remaining)))

            # All ranks should mark the pixels for this batch as completed. 'last_pixel' attribute will be updated later
            # Setting each section to 1 independently
            for curr_slice in integers_to_slices(self.__pixels_in_batch):
                self._h5_status_dset[curr_slice] = 1

            self._read_data_chunk()

        if self.verbose:
            print('Rank {} - Finished computing all jobs!'.format(self.mpi_rank))

        if self.mpi_comm is not None:
            self.mpi_comm.barrier()

        if self.mpi_rank == 0:
            print('Finished processing the entire dataset!')

        # Update the legacy 'last_pixel' attribute here:
        if self.mpi_rank == 0:
            self.h5_results_grp.attrs['last_pixel'] = self.h5_main.shape[0]

        return self.h5_results_grp


def parallel_compute(data, func, cores=1, lengthy_computation=False, func_args=None, func_kwargs=None, verbose=False):
    """
    Computes the provided function using multiple cores using the joblib library

    Parameters
    ----------
    data : numpy.ndarray
        Data to map function to. Function will be mapped to the first axis of data
    func : callable
        Function to map to data
    cores : uint, optional
        Number of logical cores to use to compute
        Default - 1 (serial computation)
    lengthy_computation : bool, optional
        Whether or not each computation is expected to take substantial time.
        Sometimes the time for adding more cores can outweigh the time per core
        Default - False
    func_args : list, optional
        arguments to be passed to the function
    func_kwargs : dict, optional
        keyword arguments to be passed onto function
    verbose : bool, optional. default = False
        Whether or not to print statements that aid in debugging
    Returns
    -------
    results : list
        List of computational results
    """
    from warnings import warn
    warn('Please call this function either as pyUSID.processing.comp_utils.parallel_compute() or simply as '
         'pyUSID.parallel_compute() instead in the future', FutureWarning)
    from .comp_utils import parallel_compute
    return parallel_compute(data, func, cores=cores, lengthy_computation=lengthy_computation, func_args=func_args,
                            func_kwargs=func_kwargs, verbose=verbose)
