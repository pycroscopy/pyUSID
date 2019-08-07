Scaling pyUSID on the SHPC Condo
================================
:Author: Emily Costa
:Created on: 08/07/2019

.. note:: If you have no prior experience in running jobs on SHPC Condo, please refer to the SHPC Condo introductory tutorials on `this GitHub repository <https://github.com/emilyjcosta5/scalable_analytics>`_.

Why use the SHPC Condo with pyUSID?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For some functions of pyUSID, parallel computing can be a helpful tool to complete a computation in a reasonable time period. As the parallel_compute() function in pyUSID does not scale up to multi-node machines, mpi4py can be used to scale computation to clusters and supercomputers for computationally heavy functions in the pyUSID and pycroscopy packages. This tutorial uses the SHPC Condo at Oak Ridge National Laboratory, but can be applied to HPC systems that use PBS files to submit and deploy jobs.

PBS Script for a Single Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When running code on a single node, MPI4py can be used, but is not necessary. 

.. note:: Make sure to run the following commands prior to running your python script in order to set up the proper environment, which includes mpi4py:
   .. code:: bash
   module load PE-intel
   module load python/3.6.3
   
Now
