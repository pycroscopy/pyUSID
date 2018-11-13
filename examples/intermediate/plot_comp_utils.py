"""
======================================================================================
12. Computing utilities
======================================================================================

**Suhas Somnath**

8/12/2017

**This is a short walk-through of useful utilities in pyUSID.processing.comp_utils that simplify common computational
tasks.**

.. tip::
    You can download and run this document as a Jupyter notebook using the link at the bottom of this page.
"""

from __future__ import print_function, division, unicode_literals
from multiprocessing import cpu_count
import subprocess
import sys


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
# Package for downloading online files:
try:
    import pyUSID as usid
except ImportError:
    print('pyUSID not found.  Will install with pip.')
    import pip
    install('pyUSID')
    import pyUSID as usid

########################################################################################################################
# recommend_cpu_cores()
# ---------------------
# Time is of the essence and every developer wants to make the best use of all available cores in a CPU for massively
# parallel computations. ``recommend_cpu_cores()`` is a popular function that looks at the number of parallel operations,
# available CPU cores, duration of each computation to recommend the number of cores that should be used for any
# computation. If the developer / user requests the use of N CPU cores, this function will validate this number against
# the number of available cores and the nature (lengthy / quick) of each computation. Unless, a suggested number of
# cores is specified, ``recommend_cpu_cores()`` will always recommend the usage of N-2 CPU cores, where N is the total
# number of logical cores (Intel uses hyper-threading) on the CPU to avoid using up all computational resources and
# preventing the computation from making the computer otherwise unusable until the computation is complete
# Here, we demonstrate this function being used in a few use cases:

print('This CPU has {} cores available'.format(cpu_count()))

########################################################################################################################
# **Case 1**: several independent computations or jobs, each taking far less than 1 second. The number of desired cores
# is not specified. The function will return 2 lesser than the total number of cores on the CPU
num_jobs = 14035
recommeded_cores = usid.processing.comp_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False)
print('Recommended number of CPU cores for {} independent, FAST, and parallel '
      'computations is {}\n'.format(num_jobs, recommeded_cores))

########################################################################################################################
# **Case 2**: Several independent and fast computations, and the function is asked if 3 cores is OK. In this case, the
# function will allow the usage of the 3 cores so long as the CPU actually has 3 or more cores
requested_cores = 3
recommeded_cores = usid.processing.comp_utils.recommend_cpu_cores(num_jobs, requested_cores=requested_cores, lengthy_computation=False)
print('Recommended number of CPU cores for {} independent, FAST, and parallel '
      'computations using the requested {} CPU cores is {}\n'.format(num_jobs, requested_cores, recommeded_cores))

########################################################################################################################
# **Case 3**: Far fewer independent and fast computations, and the function is asked if 3 cores is OK. In this case,
# configuring multiple cores for parallel computations will probably be slower than serial computation with a single
# core. Hence, the function will recommend the use of only one core in this case.
num_jobs = 13
recommeded_cores = usid.processing.comp_utils.recommend_cpu_cores(num_jobs, requested_cores=requested_cores, lengthy_computation=False)
print('Recommended number of CPU cores for {} independent, FAST, and parallel '
      'computations using the requested {} CPU cores is {}\n'.format(num_jobs, requested_cores, recommeded_cores))

########################################################################################################################
# **Case 4**: The same number of a few independent computations but eahc of these computations are expected to be
# lengthy. In this case, the overhead of configuring the CPU core for parallel computing is worth the benefit of
# parallel computation. Hence, the function will allow the use of the 3 cores even though the number of computations is
# small.
recommeded_cores = usid.processing.comp_utils.recommend_cpu_cores(num_jobs, requested_cores=requested_cores, lengthy_computation=True)
print('Recommended number of CPU cores for {} independent, SLOW, and parallel '
      'computations using the requested {} CPU cores is {}'.format(num_jobs, requested_cores, recommeded_cores))

########################################################################################################################
# get_available_memory()
# ----------------------
# Among the many best-practices we follow when developing a new data analysis or processing class is memory-safe
# computation. This handy function helps us quickly get the available memory. Note that this function returns the
# available memory in bytes. So, we have converted it to gigabytes here:
print('Available memory in this machine: {} GB'.format(usid.processing.comp_utils.get_available_memory() / 1024 ** 3))
