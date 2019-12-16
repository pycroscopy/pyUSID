# -*- coding: utf-8 -*-
"""
Abstract :class:`~pyUSID.io.translator.Translator` base-class

Created on Tue Nov  3 15:07:16 2015

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import abc
import sys
import os
from .dtype_utils import validate_list_of_strings, validate_single_string_arg
from ..processing.comp_utils import get_available_memory

if sys.version_info.major == 3:
    unicode = str
else:
    FileNotFoundError = ValueError


class Translator(object):
    """
    Abstract class that defines the most basic functionality of a data format translator.
    A translator converts experimental data from binary / proprietary
    data formats to a single standardized USID HDF5 data file
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, max_mem_mb=1024, *args, **kwargs):
        """
        Parameters
        -----------
        max_mem_mb : unsigned integer (Optional. Default = 1024)
            Maximum system memory (in megabytes) that the translator can use
            
        Returns
        -------
        Translator object
        """
        self.max_ram = min(max_mem_mb * 1024 ** 2, 0.75 * get_available_memory())

    @abc.abstractmethod
    def translate(self, *args, **kwargs):
        """
        Abstract method.
        To be implemented by extensions of this class. God I miss Java!
        """
        raise NotImplementedError('The translate method needs to be implemented by the child class')

    @staticmethod
    def is_valid_file(file_path, *args, **kwargs):
        """
        Checks whether the provided file can be read by this translator.

        This basic function compares the file extension against the "extension" keyword argument
        If the extension matches, this function returns True

        Parameters
        ----------
        file_path : str
            Path to raw data file

        Returns
        -------
        file_path : str
            Path to the file that needs to be provided to translate()
            if the provided file was indeed a valid file
            Else, None
        """
        file_path = validate_single_string_arg(file_path, 'file_name')

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path + ' does not exist')

        targ_ext = kwargs.get('extension', None)
        if not targ_ext:
            raise NotImplementedError('Either is_valid_file() has not been implemented by this translator '
                                      'or the "extension" keyword argument was missing')
        if isinstance(targ_ext, (str, unicode)):
            targ_ext = [targ_ext]
        targ_ext = validate_list_of_strings(targ_ext, parm_name='(keyword argument) "extension"')

        # Get rid of any '.' separators that may be in the list of extensions
        # Also turn to lower case for case insensitive comparisons
        targ_ext = [item.replace('.', '').lower() for item in targ_ext]

        file_path = os.path.abspath(file_path)
        extension = os.path.splitext(file_path)[1][1:]

        # Ensure extension is lower case just like targets above
        extension = extension.lower()

        if extension in targ_ext:
            return file_path
        else:
            return None
