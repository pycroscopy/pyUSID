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
from warnings import warn
import time as tm
from .io_utils import get_time_stamp
from .dtype_utils import validate_list_of_strings
from ..processing.comp_utils import get_available_memory

if sys.version_info.major == 3:
    unicode = str


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
        bool : Whether or not this translator can read this file
        """
        targ_ext = kwargs.get('extension', None)
        if not targ_ext:
            raise NotImplementedError('Either is_valid_file() has not been implemented by this translator '
                                      'or the "extension" keyword argument was missing')
        if isinstance(targ_ext, (str, unicode)):
            targ_ext = [targ_ext]
        targ_ext = validate_list_of_strings(targ_ext, parm_name='(keyword argument) "extension"')

        file_path = os.path.abspath(file_path)
        extension = os.path.splitext(file_path)[1][1:]
        if extension in targ_ext:
            return True
        else:
            return False


def generate_dummy_main_parms():
    """
    Generates a (dummy) dictionary of parameters that will be used at the root level of the h5 file

    Returns
    ----------
    main_parms : dictionary
        Dictionary containing basic descriptors that describe a dataset
    """
    warn('generate_dummy_main_parms() only serves as an example for metadata that should be included with the data.\n'
        'Please consider populating appropriate metadata manually instead of using this function', DeprecationWarning)

    main_parms = dict()
    main_parms['translate_time'] = get_time_stamp()
    main_parms['instrument'] = 'cypher_west'
    main_parms['xcams_id'] = 'abc'
    main_parms['user_name'] = 'John Doe'
    main_parms['sample_name'] = 'PZT'
    main_parms['sample_description'] = 'Thin Film'
    main_parms['project_name'] = 'Band Excitation'
    main_parms['project_id'] = 'CNMS_2015B_X0000'
    main_parms['comments'] = 'Band Excitation data'
    main_parms['data_tool'] = 'be_analyzer'
    # This parameter actually need not be a dummy and can be extracted from the parms file
    main_parms['experiment_date'] = '2015-10-05 14:55:05'
    main_parms['experiment_unix_time'] = tm.time()

    return main_parms
