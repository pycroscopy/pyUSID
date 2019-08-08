Scaling pyUSID.Process on the Compute Clusters
==============================================
:Author: Emily Costa, Suhas Somnath
:Created on: 08/07/2019

**Here we provide instructions and advice on scaling computations based on pyUSID.Process
to multiple computers in a high-performance-computing (HPC) cluster**

Introduction
------------
Certain data processing routines are very time consuming because of the sheer size of the data and/or
the computational complexity of the data processing routine.
As a consequence, it is not feasible to run such computations on personal computers.
Often, such computations are ``embarrasingly parallel`` meaning that the processing of one portion (e.g. pixel)
of data is independent from  the processing of all other portions of data.

The `pyUSID.parallel_compute() <./_autosummary/pyUSID.processing.html#pyUSID.processing.parallel_compute>`_
function can effectively distribute the computation over all available cores in a CPU and reduce the computational time.
However, ``pyUSID.parallel_compute()`` only distribute computations within a single CPU in a single personal computer.
In such cases and when available, it is recommended that such computations be run on a university / national lab
compute cluster for timely processing of the data.

The `pyUSID.Process <./auto_examples/intermediate/plot_process.html#sphx-glr-auto-examples-intermediate-plot-process-py>`_
class facilitates the formalization of data processing that needs
to be performed routinely or by  multiple researchers in a repeatable and systematic manner.
``pyUSID.Process`` has built-in mechanisms to automatically detect when it is
being called within an HPC context (as opposed to within a personal computer) and use all available
compute nodes (individual computers within a cluster) to distribute and accelerate the computation.
The user does **not** need to write any new code or restructure existing code in classes
that extend ``pyUSID.Process`` to take advantage of such capabilities.
Two examples of such classes are the
`pycroscopy.processing.SignalFilter <https://pycroscopy.github.io/pycroscopy/_autosummary/_autosummary/pycroscopy.processing.signal_filter.html#pycroscopy.processing.signal_filter.SignalFilter>`_
and `pycroscopy.analysis.GIVBayesian <https://pycroscopy.github.io/pycroscopy/_autosummary/_autosummary/pycroscopy.analysis.giv_bayesian.html#pycroscopy.analysis.giv_bayesian.GIVBayesian>`_.

HPCs are structured and operate in a manner that is different from a personal computer.
As a consequence, running the computations on a (child of) ``pyUSID.Process`` on an HPC necessitate a few small scripts.
Please read `this document <https://github.com/pycroscopy/scalable_analytics/blob/master/shpc_condo_tutorial.md>`_ to learn how to submit computational ``jobs`` in HPCs.

This tutorial uses the `CADES SHPC Condo <https://cades.ornl.gov/service-suite/scalable-hpc/>`_
at Oak Ridge National Laboratory. However, most of the instructions and guidance in this document
can be applied to other HPC systems to submit and deploy computational jobs.

This example will demonstrate how to perform signal filtering using the
`pycroscopy.processing.SignalFilter <https://pycroscopy.github.io/pycroscopy/_autosummary/_autosummary/pycroscopy.processing.signal_filter.html#pycroscopy.processing.signal_filter.SignalFilter>`_
class, on a h5USID dataset on an HPC.

Computing on a Personal Computer
--------------------------------
In order to perform certain data processing on a personal computer, one needs a
python script that opens the file with the data, assigns appropriate parameters,
and instructs the ``pycroscopy.processing.SignalFilter`` class to perform the filtering:

.. code:: python

   # Import appropriate packages:
   # To read (and write) to the data file:
   import h5py
   # To specify the parameters for filtering:
   from pycroscopy.processing.fft import LowPassFilter
   # The class that will apply the filter
   from pycroscopy.processing.signal_filter import SignalFilter

   # Specify the path to the data file:
   h5_path = 'giv_raw.h5'

   # Open the data file:
   # --------------------------------------------------------------
   # this line will need to be changed for use on HPC:
   h5_f = h5py.File(h5_path, mode='r+')
   # --------------------------------------------------------------

   # find the dataset of the interest within  the data file:
   h5_grp = h5_f['Measurement_000/Channel_000']
   h5_main = h5_grp['Raw_Data']

   # find parameters necessary for setting up the filters:
   samp_rate = h5_grp.attrs['IO_samp_rate_[Hz]']
   num_spectral_pts = h5_main.shape[1]

   # Set up the desired filter parameters:
   frequency_filters = [LowPassFilter(num_spectral_pts, samp_rate, 10E+3)]
   noise_tol = 1E-6

   # Instantiate and set up the class that will perform the filtering
   sig_filt = SignalFilter(h5_main, frequency_filters=frequency_filters,
                           noise_threshold=noise_tol, write_filtered=True,
                           write_condensed=False, num_pix=1, verbose=False)

   # Perform the filtering:
   h5_filt_grp = sig_filt.compute()

   # Don't forget to close the file
   h5_f.close()

.. note::

   Running the python script as is on an HPC without submitting a computational ``job``
   would cause the job to run on the ``login node`` rather than on the ``compute node``.
   Users are highly discouraged from running computationally intensive tasks (especially
   the computational task of interest) on the ``login node``. Users are required to
   submit ``jobs`` instead.

   Even if a job is submitted based on the above script, the computation would only
   run on a single (node) computer within the HPC cluster.

Running a job on a Single Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When running code on a single node, MPI4py can be used and is preferred, but is not necessary. We will create a python script that opens the hdf5 file, then computes on it using the SignalFilter from pycroscopy.
The following is the python script that we are going to scale to a single node on the Condo:

Now, we need to create a simple PBS file to execute the job on the SHPC Condo. The two main components of the PBS file will be (1) specifying PBS flags and (2) the main program. The following is an example PBS script, along with helpful comments:

.. code:: bash

   #!/bin/bash

   ### Set the job name. Your output files will share this name.
   #PBS -N mpiSignalFilter
   ### Enter your email address. Errors will be emailed to this address.
   #PBS -M email@ornl.gov
   ### Node spec, number of nodes and processors per node that you desire.
   ### One node and 16 cores per node in this case.
   #PBS -l nodes=1:ppn=36
   ### Tell PBS the anticipated runtime for your job, where walltime=HH:MM:S.
   #PBS -l walltime=0:00:30:0
   ### The LDAP group list they need; cades-birthright in this case.
   #PBS -W group_list=cades-ccsd
   ### Your account type. Birtright in this case.
   #PBS -A ccsd
   ### Quality of service set to burst.
   #PBS -l qos=std


   ## begin main program ##

   ### Remove old modules to ensure a clean state.
   module purge

   ### Load modules (your programming environment)
   module load PE-gnu
   ### Load custom python virtual environment
   module load python/3.6.3
   ###source /lustre/or-hydra/cades-ccsd/syz/python_3_6/bin/activate


   ### Check loaded modules
   module list

   ### Switch to the working directory (path of your PBS script).
   EGNAME=signal_filter
   DATA_PATH=$HOME/giv/pzt_nanocap_6_just_translation_copy.h5
   SCRIPTS_PATH=$HOME/mpi_tutorials/$EGNAME
   WORK_PATH=/lustre/or-hydra/cades-ccsd/syz/pycroscopy_ensemble

   cd $WORK_PATH
   mkdir $EGNAME
   cd $EGNAME

   ### Show current directory.
   pwd

   ### Copy data:
   DATA_NAME=giv_raw.h5
   rm -rf $DATA_NAME
   cp $DATA_PATH $DATA_NAME

   ### Copy python files:
   cp $SCRIPTS_PATH/fft.py .
   cp $SCRIPTS_PATH/filter.py .
   cp $SCRIPTS_PATH/gmode_utils.py .
   cp $SCRIPTS_PATH/signal_filter.py .
   cp $SCRIPTS_PATH/process.py .

   ls -hl

   ### execute code using python and add any flags you desire.
   python -m cProfile -s cumtime filter.py

Once the python and PBS scripts are set up, you can simply the following command on the SHPC Condo to submit a job:

.. code:: bash

   qsub my_pbs_script.pbs


Multiple Node Computing
-------------------
Before getting into the specifics of running on the multiple nodes, we would need to change
the way the data file is being opened. There are only two things that will need to be added to the h5py file instance:
   1. **The driver:** will map the logical HDF5 address space to a storage mechanism and we need to specify the 'mpio' file driver. This will allow mpi4py to delegate memory allocation for the HDF5 file.
   2. **Comm:** class for communication of generic Python objects

from:

.. code:: python

   h5_f = h5py.File(h5_path, mode='r+')

to:

.. code:: python

   from mpi4py import MPI
   h5_f = h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD)

The above modification asks ``h5py`` to open the data  file in such a manner that
multiple python processes running on multiple compute ``nodes`` (individual computers within the HPC)
can read and write to the data file in parallel and independently.

In order to distribute the same computation on multiple nodes within a compute cluster,
one would need to submit a computational job in addition to making minor edits to the
python script above.


In this example, we will use mpiexec to initialize a parallel job from within the PBS batch. Mpiexec uses the task manager library of PBS to spawn copies of the executable on the nodes in a PBS allocation.

.. note:: Make sure to run the following commands prior to running your python script:

       module load PE-intel

       module load python/3.6.3
  
   Now, your programming environment is setup and includes mpi4py.

The following is an example of a script that runs a signal filter through a USID dataset using pycroscopy, a package built on pyUSID, using a multiple node remote machine (in this case, CADES SHPC Condo).

Now, time to build the PBS script for multiple nodes. We add a few components to the execution command:
   1. **mpiexec** 
       to run an mpi program.
   2. **--map-by ppr:1:node** 
       **ppr** stands for processes per resource. 

       **ppr:N:resource** assigns N processes to each resource of type resource available on the host. In the case of the Condo, the resource is 'node'.

.. code:: bash
   #!/bin/bash
   
   ### Set the job name. Your output files will share this name.
   #PBS -N mpiSignalFilter
   ### Enter your email address. Errors will be emailed to this address.
   #PBS -M email@ornl.gov
   ### Node spec, number of nodes and processors per node that you desire.
   ### One node and 16 cores per node in this case.
   #PBS -l nodes=2:ppn=36
   ### Tell PBS the anticipated runtime for your job, where walltime=HH:MM:S.
   #PBS -l walltime=0:00:30:0
   ### The LDAP group list they need; cades-birthright in this case.
   #PBS -W group_list=cades-ccsd
   ### Your account type. Birtright in this case.
   #PBS -A ccsd
   ### Quality of service set to burst.
   #PBS -l qos=std


   ## begin main program ##

   ### Remove old modules to ensure a clean state.
   module purge

   ### Load modules (your programming environment)
   module load PE-gnu
   ### Load custom python virtual environment
   module load python/3.6.3
   ###source /lustre/or-hydra/cades-ccsd/syz/python_3_6/bin/activate


   ### Check loaded modules 
   module list

   ### Switch to the working directory (path of your PBS script).
   EGNAME=signal_filter
   DATA_PATH=$HOME/giv/pzt_nanocap_6_just_translation_copy.h5
   SCRIPTS_PATH=$HOME/mpi_tutorials/$EGNAME
   WORK_PATH=/lustre/or-hydra/cades-ccsd/syz/pycroscopy_ensemble

   cd $WORK_PATH
   mkdir $EGNAME
   cd $EGNAME

   ### Show current directory.
   pwd

   ### Copy data:
   DATA_NAME=giv_raw.h5
   rm -rf $DATA_NAME
   cp $DATA_PATH $DATA_NAME

   ### Copy python files:
   cp $SCRIPTS_PATH/fft.py .
   cp $SCRIPTS_PATH/filter_mpi.py .
   cp $SCRIPTS_PATH/gmode_utils.py .
   cp $SCRIPTS_PATH/mpi_signal_filter.py .
   cp $SCRIPTS_PATH/mpi_process.py .

   ls -hl

   ### MPI run followed by the name/path of the binary.
   mpiexec --map-by ppr:1:node python -m cProfile -s cumtime filter_mpi.py

FAQs
~~~~

Why use the SHPC Condo with pyUSID?
###################################
For some functions of pyUSID, parallel computing can be a helpful tool to complete a computation in a reasonable time period. As the parallel_compute() function in pyUSID does not scale up to multi-node machines, mpi4py can be used to scale computation to clusters and supercomputers for computationally heavy functions in the pyUSID and pycroscopy packages. This tutorial uses the SHPC Condo at Oak Ridge National Laboratory, but can be applied to HPC systems that use PBS files to submit and deploy jobs.

Why mpiexec instead of mpirun?
##############################
Reasons to use mpiexec rather than a script (mpirun) or an external daemon (mpd):

   1. Starting tasks with the TM interface is much faster than invoking a separate rsh or ssh once for each process.
   2. Resources used by the spawned processes are accounted correctly with mpiexec, and reported in the PBS logs, because all the processes of a parallel job remain under the control of PBS, unlike when using startup scripts such as mpirun.
   3. Tasks that exceed their assigned limits of CPU time, wallclock time, memory usage, or disk space are killed cleanly by PBS. It is quite hard for processes to escape control of the resource manager when using mpiexec.
   4. You can use mpiexec to enforce a security policy. If all jobs are required to startup using mpiexec and the PBS execution environment, it is not necessary to enable rsh or ssh access to the compute nodes in the cluster.

Reference: https://www.osc.edu/~djohnson/mpiexec/ 

Why is MPI used in both the Python and PBS script?
##################################################
**Python script** is where MPI is used for point-to-point (sends, receives), and collective (broadcasts, scatters, gathers) communications of any picklable Python object.

**PBS script** is where the command is put to start the parallel job. In our case, mpiexec starts the program a specfied number of times in parallel, forming a parallel job.

Who do I contact if I am struggling to run a job?
#################################################
Contact CADES user support team at cades-help@ornl.gov or join the CADES Slack channel at https://cades.slack.com/signup

For help with pyUSID and/or pycroscopy, contact our team at `this email <pycroscopy@gmail.com>`_
