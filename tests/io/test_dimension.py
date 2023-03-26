# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import unittest
import numpy as np

from pyUSID.io import dimension

if sys.version_info.major == 3:
    unicode = str


class TestDimension(unittest.TestCase):

    def test_values_as_array(self):
        name = 'Bias'
        units = 'V'
        values = np.random.rand(5)

        descriptor = dimension.Dimension(name, units, values)
        for expected, actual in zip([name, units, values],
                                    [descriptor.name, descriptor.units, descriptor.values]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

    def test_values_as_length(self):
        name = 'Bias'
        units = 'V'
        values = np.arange(5)

        descriptor = dimension.Dimension(name, units, len(values))
        for expected, actual in zip([name, units],
                                    [descriptor.name, descriptor.units]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))
        self.assertTrue(np.allclose(values, descriptor.values))

    def test_repr(self):
        name = 'Bias'
        quantity = 'generic'
        units = 'V'
        values = np.arange(5, dtype=float)

        descriptor = dimension.Dimension(name, units, len(values))
        print(type(descriptor))
        actual = '{}'.format(descriptor)
        expected = '{}: {} ({}) mode:{} : {}'.format(name, quantity, units, descriptor.mode, values)
        self.assertEqual(actual, expected)

    def test_equality(self):
        name = 'Bias'
        units = 'V'

        dim_1 = dimension.Dimension(name, units, [0, 1, 2, 3, 4])
        dim_2 = dimension.Dimension(name, units, np.arange(5, dtype=np.float32))
        self.assertEqual(dim_1, dim_2)

    def test_inequality(self):
        name = 'Bias'
        units = 'V'
        values = [0, 1, 2, 3]

        left = dimension.Dimension(name, units, values)
        right = dimension.Dimension(name, units, [0, 1, 2, 4])
        self.assertFalse(left == right)

        left = dimension.Dimension(name, units, [0, 1, 2])
        right = dimension.Dimension(name, units, values)
        self.assertFalse(left == right)

        left = dimension.Dimension('name', units, values)
        right = dimension.Dimension(name, units, values)
        self.assertFalse(left == right)

        left = dimension.Dimension(name, 'units', values)
        right = dimension.Dimension(name, units, values)
        self.assertFalse(left == right)

        left = dimension.Dimension(name, units, values,
                                     mode=dimension.DimType.DEPENDENT)
        right = dimension.Dimension(name, units, values)
        self.assertFalse(left == right)

    def test_invalid_mode(self):
        with self.assertRaises(TypeError):
            _ = dimension.Dimension('Name', 'units', 5, mode='Incomplete')

    def test_default_mode(self):
        dim = dimension.Dimension('Name', 'units', 1)
        self.assertEqual(dim.mode, dimension.DimType.DEFAULT)

    def test_illegal_instantiation(self):

        with self.assertRaises(TypeError):
            _ = dimension.Dimension('Name', 14, np.arange(4))

        with self.assertRaises(TypeError):
            _ = dimension.Dimension(14, 'nm', np.arange(4))

        with self.assertRaises(ValueError):
            _ = dimension.Dimension('Name', 'unit', 0)

        with self.assertRaises(TypeError):
            _ = dimension.Dimension('Name', 'unit', 'invalid')


class TestDimType(unittest.TestCase):

    def test_dim_type_invalid_comparison(self):
        with self.assertRaises(TypeError):
            dimension.DimType.INCOMPLETE == "Default"

    def test_dim_type_valid_comparison(self):
        self.assertTrue(dimension.DimType.DEFAULT < dimension.DimType.INCOMPLETE)
        self.assertTrue(dimension.DimType.INCOMPLETE < dimension.DimType.DEPENDENT)
