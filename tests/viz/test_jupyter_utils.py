"""
Created on Thurs Jun  27 2019

@author: Emily Costa
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import os
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from pyUSID.io.write_utils import Dimension
from pyUSID.viz.jupyter_utils import simple_ndim_visualizer, save_fig_filebox_button

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


class TestSimpleNdimVisualizer(unittest.TestCase):

    """
    def test_correct(self):
        data_mat = np.random.rand(2,3,5,7)
        x = np.arange(2)
        y = np.arange(3)
        z = np.arange(5)
        w = np.arange(7)
        pos_dims = [Dimension('X','unit',x), Dimension('Y','unit',y)]
        spec_dims = [Dimension('Z','unit',z), Dimension('W','unit',w)]
        simple_ndim_visualizer(data_mat, pos_dims, spec_dims)
    """

    def test_not_iterable(self):
        arr = np.arange(100).reshape(10, 10)
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
    """
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
    """


class TestSaveFigFileboxButton(unittest.TestCase):

    pass

    """
    def test_correct(self):
        fig, axis = plt.subplots()
        file_name = 'filename'
        save_fig_filebox_button(fig,file_name)
    """


if __name__ == '__main__':
    unittest.main()
