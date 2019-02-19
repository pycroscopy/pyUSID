"""
======================================================================================
11. Input / Output utilities
======================================================================================

**Suhas Somnath**

8/12/2017

**This is a short walk-through of useful utilities in pyUSID.io.io_utils that simplify common i/o tasks.**

.. tip::
    You can download and run this document as a Jupyter notebook using the link at the bottom of this page.
"""

from __future__ import print_function, division, unicode_literals
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
# String formatting utilities
# ===========================
# Frequently, there is a need to print out logs on the console to inform the user about the size of files, or estimated
# time remaining for a computation to complete, etc. pyUSID.io_utils has a few handy functions that help in
# formatting quantities in a human readable format.
#
# format_size()
# --------------
# One function that uses this functionality to print the size of files etc. is format_size(). While one can manually
# print the available memory in gibibytes (see above), ``format_size()`` simplifies this substantially:
mem_in_bytes = usid.processing.comp_utils.get_available_memory()
print('Available memory in this machine: {}'.format(usid.io_utils.format_size(mem_in_bytes)))

########################################################################################################################
# format_time()
# -------------
# On the same lines, ``format_time()`` is another handy function that is great at formatting time and is often used in
# Process and Fitter to print the remaining time
print('{} seconds = {}'.format(14497.34, usid.io_utils.format_time(14497.34)))

########################################################################################################################
# format_quantity()
# -----------------
# You can generate your own formatting function based using the generic function: ``format_quantity()``.
# For example, if ``format_time()`` were not available, we could get the same functionality via:
units = ['msec', 'sec', 'mins', 'hours']
factors = [0.001, 1, 60, 3600]
time_value = 14497.34
print('{} seconds = {}'.format(14497.34, usid.io_utils.format_quantity(time_value, units, factors)))

########################################################################################################################
# formatted_str_to_number()
# -------------------------
# pyUSID also has a handy function for the inverse problem of getting a numeric value from a formatted string:
unit_names = ["MHz", "kHz"]
unit_magnitudes = [1E+6, 1E+3]
str_value = "4.32 MHz"
num_value = usid.io_utils.formatted_str_to_number(str_value, unit_names, unit_magnitudes, separator=' ')
print('formatted_str_to_number says: {} = {}'.format(str_value, num_value))

########################################################################################################################
# get_time_stamp()
# ----------------
# We try to use a standardized format for storing time stamps in HDF5 files. The function below generates the time
# as a string that can be easily parsed if need be
print('Current time is: {}'.format(usid.io_utils.get_time_stamp()))

########################################################################################################################
# Communication utilities
# ========================
# file_dialog()
# -------------
# This handy function generates a file window to select files. We encourage you to try this function out since it cannot
# demonstrated within this static document.
#
# check_ssh()
# -----------
# When developing workflows that need to work on remote or virtual machines in addition to one's own personal computer
# such as a laptop, this function is handy at letting the developer know where the code is being executed

print('Running on remote machine: {}'.format(usid.io_utils.check_ssh()))
