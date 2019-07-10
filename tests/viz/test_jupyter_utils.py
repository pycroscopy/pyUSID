"""
Created on Thurs Jun  27 2019

@author: Emily Costa
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import numpy as np
from pyUSID.viz.jupyter_utils import simple_ndim_visualizer

class TestSimpleNdimVisualizer(unittest.TestCase):

    def test_not_iterable(self):
        arr = np.arange(100).reshape(10,10)
        num = 1
        with self.assertRaises(TypeError):
            simple_ndim_visualizer(arr, num, num)

    def test_not_dimension_type(self):
        arr = np.arange(100).reshape(10, 10)
        list = [1, 2, 3]
        with self.assertRaises(TypeError):
            simple_ndim_visualizer(arr, list, list)

    def test_too_many_dims(self):
        arr = np.arange(100).reshape(10, 10)
        list = [1, 2, 3]
        with self.assertRaises(TypeError):
            simple_ndim_visualizer(arr, list, list)

    def test_too_few_dims(self):
        pass

    def test_not_matching_dims_sizes(self):
        pass

    def test_not_matching_dims(self):
        pass

    def test_item_val_none(self):
        pass

    def test_item_val_not_equals_data_shape(self):
        pass

    def test_completed_viz_xdim_none(self):
        pass

    def test_completed_viz_xdim_not_none(self):
        pass

if __name__ == '__main__':
    unittest.main()
