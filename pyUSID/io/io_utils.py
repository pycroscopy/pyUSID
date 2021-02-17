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
