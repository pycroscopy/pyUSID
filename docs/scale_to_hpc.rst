Scaling to High-Performance-Computing Clusters
==============================================
:Author: Suhas Somnath, Emily Costa
:Created on: 08/20/2019

**Here we provide instructions and advice on scaling computations based on pyUSID.Process
to multiple computers in a high-performance-computing (HPC) cluster**

Introduction
------------
Certain data processing routines are very time consuming because of the sheer size of the data and/or
the computational complexity of the data processing routine.
Often, such computations are ``embarrasingly parallel`` meaning that the processing of one portion (e.g. pixel)
of data is independent from  the processing of all other portions of data.

The `pyUSID.parallel_compute() <./_autosummary/pyUSID.processing.html#pyUSID.processing.parallel_compute>`_
function can effectively distribute the computation over all available cores in a CPU and reduce the computational time.
However, ``pyUSID.parallel_compute()`` only distribute computations within a single CPU in a single personal computer.
As a consequence, it may not be feasible to run large / lengthy computations on personal computers.
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

.. note::

   The changes necessary to run on a HPC machine are far less intimidating than they appear!

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
   run on a **single** (node) computer within the HPC cluster.

Computing on an HPC
-------------------
The fundamental change in scaling from a personal computer to an HPC is the communication
of instructions and data between the multiple computers within an HPC so that the
multiple python processes spawned on the individual computers on an HPC can work
together to reduce the computational time. Most HPC code uses an inter-node
communication paradigm called the ``message passing interface (MPI)``. ``mpi4py``
is the python API for interacting with the MPI on the HPC. Note that in-depth
knowledge of MPI or mpi4py is unnecessary for understanding this tutorial since
``pyUSID.Process`` handles most of the heavy lifting behind the scenes.

Below are the changes necessary to scale from a personal computer to an HPC:

1. Changing the HDF5 file access mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We need to tell ``h5py`` to open the data  file in such a manner that
multiple python processes running on multiple compute ``nodes`` (individual computers within the HPC)
can read and write to the data file in parallel and independently:

from:

.. code:: python

   h5_f = h5py.File(h5_path, mode='r+')

to:

.. code:: python

   from mpi4py import MPI
   h5_f = h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD)

Here:

1. **driver:** will map the logical HDF5 address space to a storage mechanism and
   we need to specify the '`mpio'` file driver. This will allow mpi4py to delegate
   memory allocation for the HDF5 file.
2. **comm:** class for communication of generic Python objects

.. note::

   It is best to have a single version of a script that works on both laptops and
   HPC clusters. The following modification would allow the script to adapt either to
   a personal computer or a HPC environment:

   .. code:: python

      """
      This handy function in pyUSID.comp_utils returns the MPI object if both mpi4py was
      available and if the script was called via mpirun or mpiexec. If either
      conditions fail, it returns None (e.g. - personal computer)
      """
      MPI = usid.comp_utils.get_MPI()

      # At a minimum, we want to read the file in an editable manner
      h5_kwargs = {'mode': 'r+'}

      # If the script is being called in the mpirun or mpiexec context:
      if MPI is not None:
          # Then, add the driver and communication configurations to the keyword arguments
          h5_kwargs.update({'driver': 'mpio', 'comm': MPI.COMM_WORLD})

      # Now, the file can be opened with the appropriate keyword arguments preconfigured
      h5_f = h5py.File(input_data_path, **h5_kwargs)

.. note::

   We are still not yet ready to actually run the script even though it is ready.
   See the next step.

2. Job script
~~~~~~~~~~~~~
The above modification to the main python script is in theory sufficient to run on
multiple computers in a cluster. However, most HPC clusters are not operated by a single user
and are in fact shared by multiple users unlike a personal computer.
On an HPC, the computational jobs from multiple users are handled by a ``scheduler``
that maintains queue(s) where users can request the scheduler to run their job.
Users need to request the scheduler to run their computational task by submitting a
``job script`` with appropriate information. This is the second and final part of the puzzle
when it comes to running computations on a HPC cluster.

Different HPC systems have different schedulers which expect the job script to be configured
in a specific manner. However, the basic components remain the same:

1. Details regarding the job - number of nodes, CPU processors / GPUs within each node,
   name of the user requesting the job, how long the nodes need to be reserved for the computation, etc.
2. ``Modules`` - Modules can be thought of as drivers and software libraries.
3. Setting up the script and necessary data files
4. Running the script

The following is an example PBS script, configured for the ORNL CADES SHPC Condo, along with helpful comments:

.. code:: bash

   #!/bin/bash

   ### 1. Job description

   ### Comments in this section need to be preceded by three hash symbols
   ### The scheduler reads text following a single hash symbol
   ### Set the job name. Your output files will share this name.
   #PBS -N mpiSignalFilter
   ### Enter your email address. Errors will be emailed to this address.
   #PBS -M your_email@institution.gov
   ### Number of nodes and processors per node that you desire.
   ### Two nodes each with 36 cores per node in this case.
   #PBS -l nodes=2:ppn=36
   ### Anticipated runtime for your job specified as HH:MM:S.
   ### See notes below on setting an appropriate wall-time
   #PBS -l walltime=0:00:30:0
   ### The organization / group that you belong to
   #PBS -W group_list=cades-birthright
   ### Your account type
   #PBS -A birthright
   ### Quality of service - leave this as is
   #PBS -l qos=std


   ###  2. Set up modules ##

   ### Remove old modules to ensure a clean state.
   module purge
   ### Load the programming environment
   module load PE-intel
   ### Load the python module with the appropriate packages
   module load python/3.6.3
   ### Check loaded modules
   module list

   ### 2.5 Set any environment variables here:
   ### Here we are using an Intel programming environment, so:
   ### Forcing MKL to use 1 thread only:
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1

   ### 3. Set up script and data

   # Here, we assume that the code and the data are on a fast scratch file system
   # Lustre in this case:
   cd /lustre/or-hydra/cades-ccsd/syz/signal_filter
   # Sanity check - make sure all the necessary files are in the working folder:
   ls -ahl

   ### 4. Run the script

   # More details on the flags below
   mpiexec -use-hwthread-cpus python filter_script.py

Wall time
^^^^^^^^^
The scheduler will kill the computational job once the elapsed time is greater than
the wall time requested in the job script. Besides the incompleteness of the desired
computation, this can also result in the corruption of output files if the job was killed
while some files were being modified.

It is recommended that the ``wall time`` be comfortably larger than the expected
computational time. Often, one may not know how long the computation takes and this can be
a challenge. Users are recommended to ``checkpoint`` (save intermediate or partial results)
regularly so that only a portion of the computation is lost.

.. note::

   ``pyUSID.Process`` has built-in mechanisms to ``checkpoint`` regularly and even
   restart from partially completed computations (either on laptops or on HPC clusters).
   Besides loading the parameters and providing handles to the necessary HD5 datasets,
   the user need not do anything additional to enable checkpointing in their ``Process`` class.

Queues and organizations
^^^^^^^^^^^^^^^^^^^^^^^^
The nodes in most HPC clusters are not homogeneous meaning that certain nodes may
have GPUs, more memory, more CPU cores, etc. while others may not. Often, this is
a result of upgrades / additions every few months or years with slightly different hardware
compared to the original set of nodes. Typically, the scheduler has separate queues
for each kind of nodes. One can specify which kinds of nodes to use using ``directives``.

.. note::

   This is mostly relevant only to ORNL CADES SHPC users - all ORNL users with UCAMS / XCAMS
   accounts have access to the ``CADES Birthright`` allocation. Certain divisions / groups such as
   CCSD, BSD, CNMS have their own compute hardware and queues. If you belong to any divisions
   listed `here <https://support.cades.ornl.gov/user-documentation/_book/condos/how-to-use/resource-queues.html>`_,
   you are recommended to change the ``PBS -W group_list`` and ``PBS -A`` flags.

Modules
^^^^^^^
One is recommended to clear the modules before loading them since we do not always know what modules
were already loaded. Modules are not always interchangeable. For example, the python module above
may not work (at all or as well) with another programming environment. In the above example,
all the necessary software was already available within the two modules.

HPC File systems
^^^^^^^^^^^^^^^^
Most HPC systems are connected to a slower file system (typically a network file system (NFS))
with the user's home directory and a much faster file system (typically something like ``GPFS``
or ``Lustre``) for scratch space where the raw and intermediate data directly interacting with
the compute nodes reside. It is **highly** recommended that the scripts, and data reside in the
scratch space file system to take advantage of the speed.

.. note::

   In most HPC systems, the file systems are ``purged`` every few weeks or months.
   In other words, files that have not been used in the last few weeks or months will
   be permanently deleted. Check with specific documentation.

Running the script
^^^^^^^^^^^^^^^^^^
``mpiexec`` was used to initialize a parallel job from within the scheduler batch.
``mpiexec`` uses the task manager library of PBS to spawn copies of the executable
on the nodes in a PBS allocation.

3. Submitting the job
~~~~~~~~~~~~~~~~~~~~~
Once the python script and the job script are prepared, the job can be submitted to the
scheduler via:

.. code:: bash

   qsub my_pbs_script.pbs

FAQs
----

Why mpiexec instead of mpirun?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`Reasons to use <https://www.osc.edu/~djohnson/mpiexec/>`_ ``mpiexec`` rather than a ``mpirun`` or an external daemon (``mpd``):

1. Starting tasks with the TM interface is much faster than invoking a separate rsh or ssh once for each process.
2. Resources used by the spawned processes are accounted correctly with mpiexec, and reported in the PBS logs, because all the processes of a parallel job remain under the control of PBS, unlike when using startup scripts such as mpirun.
3. Tasks that exceed their assigned limits of CPU time, wallclock time, memory usage, or disk space are killed cleanly by PBS. It is quite hard for processes to escape control of the resource manager when using mpiexec.
4. You can use mpiexec to enforce a security policy. If all jobs are required to startup using mpiexec and the PBS execution environment, it is not necessary to enable rsh or ssh access to the compute nodes in the cluster.

Reference:

Why is MPI used in both the Python and PBS script?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Python script**: ``MPI`` is used for point-to-point (``send``, ``receive``), and collective (``broadcast``, ``scatter``, ``gather``) communications of any ``pickle``-able Python object.
* **Job script**: ``mpiexec`` starts the parallel job - starts the program a specified number of times in parallel, forming a parallel job.
