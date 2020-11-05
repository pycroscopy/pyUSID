# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:07:16 2018

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import os
import sys
import h5py
import numpy as np

from . import data_utils

sys.path.append("../../pyUSID/")
from pyUSID.io.hdf_utils import get_attr
from pyUSID.io import reg_ref


if sys.version_info.major == 3:
    unicode = str


class TestRegRef(unittest.TestCase):

    def setUp(self):
        data_utils.make_beps_file()

    def tearDown(self):
        data_utils.delete_existing_file(data_utils.std_beps_path)

    def test_copy_reg_ref_reduced_dim(self):
        # TODO: Fill this test in at earliest convenience. Overriden temporarily
        assert True


if __name__ == '__main__':
    unittest.main()
