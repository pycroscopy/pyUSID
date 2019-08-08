Scaling pyUSID on the SHPC Condo
================================
:Author: Emily Costa
:Created on: 08/07/2019

The purpose of this tutorial is to provide examples and advice on running jobs on the SHPC Condo that use the pyUSID package. The SHPC Condo is managed by CADES at Oak Ridge National Laboratory. 

In our example, we will run a signal filter, which is built into the pycroscopy package, on a USID dataset, which is manipulated by the pyUSID package. 

.. note:: If you have no prior experience in running jobs on SHPC Condo, please refer to the SHPC Condo introductory tutorials on `this GitHub repository <https://github.com/emilyjcosta5/scalable_analytics>`_.

PBS Script for a Single Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When running code on a single node, MPI4py can be used, but is not necessary. We will create a python script that opens the hdf5 file, then computes on it using the SignalFilter from pycroscopy.
The following is the python script that we are going to scale to a single node on the Condo:

.. code:: python
   # filter.py
   import h5py
   from mpi4py import MPI
   from fft import LowPassFilter
   from mpi_signal_filter import SignalFilter
   
   # find the hdf5 file and open it
   h5_path = 'giv_raw.h5'
   ###################################################
   # note: this is the only line we will change for our 
   # mpi version of the python script
   h5_f = h5py.File(h5_path, mode='r+')
   ####################################################

   # find the main dataset of the file
   h5_grp = h5_f['Measurement_000/Channel_000']
   h5_main = h5_grp['Raw_Data']

   # find some needed attributes
   samp_rate = h5_grp.attrs['IO_samp_rate_[Hz]']
   num_spectral_pts = h5_main.shape[1]
   
   # create some noise
   frequency_filters = [LowPassFilter(num_spectral_pts, samp_rate, 10E+3)]
   noise_tol = 1E-6
   
   # create the object and compute
   sig_filt = SignalFilter(h5_main, frequency_filters=frequency_filters,
                           noise_threshold=noise_tol, write_filtered=True,
                           write_condensed=False, num_pix=1, verbose=False)
   h5_filt_grp = sig_filt.compute()
   
   # make sure the close the file
   h5_f.close()

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

PBS Script for Multiple Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this example, we will use mpiexec to initialize a parallel job from within the PBS batch. Mpiexec uses the task manager library of PBS to spawn copies of the executable on the nodes in a PBS allocation.

.. note:: Make sure to run the following commands prior to running your python script:

       module load PE-intel

       module load python/3.6.3
  
   Now, your programming environment is setup and includes mpi4py.

The following is an example of a script that runs a signal filter through a USID dataset using pycroscopy, a package built on pyUSID, using a multiple node remote machine (in this case, CADES SHPC Condo).

Prior to making our new MPI-aware PBS script, we will need to create a MPI version of our Python script. There are only two things that will need to be added to the h5py file instance:
   1. **The driver:** will map the logical HDF5 address space to a storage mechanism and we need to specify the 'mpio' file driver. This will allow mpi4py to delegate memory allocation for the HDF5 file.
   2. **Comm:** class for communication of generic Python objects

The Python script that MPI will execute is the following:

.. code:: python
   #mpi_filter.py
   import h5py
   from mpi4py import MPI
   from fft import LowPassFilter
   from mpi_signal_filter import SignalFilter

   h5_path = 'giv_raw.h5'
   ###################################################
   # Note: this is our changed, mpi-aware code.
   h5_f = h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD)
   ###################################################

   h5_grp = h5_f['Measurement_000/Channel_000']
   h5_main = h5_grp['Raw_Data']

   samp_rate = h5_grp.attrs['IO_samp_rate_[Hz]']
   num_spectral_pts = h5_main.shape[1]

   frequency_filters = [LowPassFilter(num_spectral_pts, samp_rate, 10E+3)]
   noise_tol = 1E-6

   sig_filt = SignalFilter(h5_main, frequency_filters=frequency_filters,
                           noise_threshold=noise_tol, write_filtered=True,
                           write_condensed=False, num_pix=1, verbose=False)
   h5_filt_grp = sig_filt.compute()

   h5_f.close()

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
