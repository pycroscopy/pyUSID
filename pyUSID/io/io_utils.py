# -*- coding: utf-8 -*-
"""
Utilities for formatting strings and other input / output methods

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn


from sidpy.io import interface_utils as iut
from sidpy.base import string_utils as sut


__all__ = ['get_time_stamp', 'file_dialog', 'format_quantity', 'format_time',
           'format_size']


def check_ssh():
    """
    Checks whether or not the python kernel is running locally (False) or remotely (True)

    Returns
    -------
    output : bool
        Whether or not the kernel is running over SSH (remote machine)
    """
    warn('pyUSID.io.io_utils.check_ssh has been moved to '
         'sidpy.io.interface_utils.check_ssh. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return iut.check_ssh()


def file_dialog(file_filter='H5 file (*.h5)', caption='Select File'):
    """
    Presents a File dialog used for selecting the .mat file
    and returns the absolute filepath of the selecte file\n

    Parameters
    ----------
    file_filter : String or list of strings
        file extensions to look for
    caption : (Optional) String
        Title for the file browser window

    Returns
    -------
    file_path : String
        Absolute path of the chosen file
    """
    warn('pyUSID.io.io_utils.file_dialog has been moved to '
         'sidpy.io.interface_utils.file_dialog. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return iut.file_dialog(file_filter=file_filter, caption=caption)


def get_time_stamp():
    """
    Teturns the current date and time as a string formatted as:
    Year_Month_Dat-Hour_Minute_Second

    Parameters
    ----------

    Returns
    -------
    String
    """
    warn('pyUSID.io.io_utils.get_time_stamp has been moved to '
         'sidpy.base.string_utils.get_time_stamp. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sut.get_time_stamp()


def format_quantity(value, unit_names, factors, decimals=2):
    """
    Formats the provided quantity such as time or size to appropriate strings

    Parameters
    ----------
    value : number
        value in some base units. For example - time in seconds
    unit_names : array-like
        List of names of units for each scale of the value
    factors : array-like
        List of scaling factors for each scale of the value
    decimals : uint, optional. default = 2
        Number of decimal places to which the value needs to be formatted

    Returns
    -------
    str
        String with value formatted correctly
    """
    warn('pyUSID.io.io_utils.format_quantity has been moved to '
         'sidpy.base.string_utils.format_quantity. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sut.format_quantity(value, unit_names, factors, decimals=decimals)


def format_time(time_in_seconds, decimals=2):
    """
    Formats the provided time in seconds to seconds, minutes, or hours

    Parameters
    ----------
    time_in_seconds : number
        Time in seconds
    decimals : uint, optional. default = 2
        Number of decimal places to which the time needs to be formatted

    Returns
    -------
    str
        String with time formatted correctly
    """
    warn('pyUSID.io.io_utils.format_time has been moved to '
         'sidpy.base.string_utils.format_time. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sut.format_time(time_in_seconds, decimals=decimals)


def format_size(size_in_bytes, decimals=2):
    """
    Formats the provided size in bytes to kB, MB, GB, TB etc.

    Parameters
    ----------
    size_in_bytes : number
        size in bytes
    decimals : uint, optional. default = 2
        Number of decimal places to which the size needs to be formatted

    Returns
    -------
    str
        String with size formatted correctly
    """
    warn('pyUSID.io.io_utils.format_size has been moved to '
         'sidpy.base.string_utils.format_size. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sut.format_size(size_in_bytes, decimals=decimals)


def formatted_str_to_number(str_val, magnitude_names, magnitude_values,
                            separator=' '):
    """
    Takes a formatted string like '4.32 MHz' to 4.32 E+6

    Parameters
    ----------
    str_val : str / unicode
        String value of the quantity. Example '4.32 MHz'
    magnitude_names : Iterable
        List of names of units like ['seconds', 'minutes', 'hours']
    magnitude_values : Iterable
        List of values (corresponding to magnitude_names) that scale the
        numeric value. Example [1, 60, 3600]
    separator : str / unicode, optional. Default = ' ' (space)
        The text that separates the numeric value and the units.

    Returns
    -------
    number
        Numeric value of the string
    """
    warn('pyUSID.io.io_utils.formatted_str_to_number has been moved to '
         'sidpy.base.string_utils.formatted_str_to_number. This copy in pyUSID will'
         'be removed in future release. Please update your import statements',
         FutureWarning)
    return sut.formatted_str_to_number(str_val, magnitude_names,
                                       magnitude_values, separator=separator)
