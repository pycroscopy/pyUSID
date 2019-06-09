# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
sys.path.append("../../pyUSID/")
from pyUSID.io import Translator


class TestIsValidFile(unittest.TestCase):

    def test_no_extension_kwarg(self):
        tran = Translator()
        with self.assertRaises(NotImplementedError):
           _ = tran.is_valid_file('blah.txt')

    def test_single_ext(self):
        tran = Translator()
        self.assertTrue(tran.is_valid_file('blah.txt', extension='txt'))

    def test_case_insensitive(self):
        tran = Translator()
        self.assertTrue(tran.is_valid_file('blah.TXT', extension='txt'))
        self.assertTrue(tran.is_valid_file('blah.txt', extension='TXT'))

    def test_with_dot_in_ext(self):
        tran = Translator()
        self.assertTrue(tran.is_valid_file('blah.txt', extension='.txt'))

    def test_multi_ext(self):
        tran = Translator()
        self.assertTrue(tran.is_valid_file('blah.txt', extension=['txt', '.png']))

    def test_diff_extension(self):
        tran = Translator()
        self.assertFalse(tran.is_valid_file('blah.txt', extension=['jpeg', '.png']))

    def test_wrong_type(self):
        tran = Translator()
        with self.assertRaises(TypeError):
            self.assertFalse(tran.is_valid_file({'hello': 3}, extension=['jpeg', '.png']))
        with self.assertRaises(TypeError):
            self.assertFalse(tran.is_valid_file('blah.txt', extension=[14, '.png']))

    def test_folder(self):
        tran = Translator()
        self.assertFalse(tran.is_valid_file('some/path/to/a/folder', extension='.txt'))


class TestTranslateNotImplemented(unittest.TestCase):

    def test_empty_translator(self):
        tran = Translator()
        with self.assertRaises(NotImplementedError):
            tran.translate('blah.txt')


if __name__ == '__main__':
    unittest.main()