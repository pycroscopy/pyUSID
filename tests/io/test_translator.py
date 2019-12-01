# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest
import sys
import os
sys.path.append("../../pyUSID/")
from pyUSID.io import Translator


class TestIsValidFile(unittest.TestCase):

    def setUp(self):

        class DummyTranslator(Translator):
            def translate(self, *args, **kwargs):
                pass

        self.translator = DummyTranslator()

        self.file_path = os.path.abspath('blah.txt')

        with open(self.file_path, mode='w') as file_handle:
            file_handle.write('Nothing')

    def tearDown(self):
        os.remove(self.file_path)

    def test_file_does_not_exist(self):
        err_type = ValueError
        if sys.version_info.major == 3:
            err_type = FileNotFoundError
        with self.assertRaises(err_type):
            _ = self.translator.is_valid_file('dfdfd.txt', extension='.txt')

    def test_no_extension_kwarg(self):
        with self.assertRaises(NotImplementedError):
           _ = self.translator.is_valid_file(self.file_path)

    def test_single_ext(self):
        self.assertEqual(self.translator.is_valid_file(self.file_path,
                                                      extension='txt'),
                         self.file_path)

    def test_case_insensitive(self):
        # self.assertTrue(self.translator.is_valid_file('blah.TXT',
        #                                               extension='txt'))
        self.assertEqual(self.translator.is_valid_file(self.file_path,
                                                      extension='TXT'),
                         self.file_path)

    def test_with_dot_in_ext(self):
        self.assertEqual(self.translator.is_valid_file(self.file_path,
                                                      extension='.txt'),
                         self.file_path)

    def test_multi_ext(self):
        self.assertEqual(self.translator.is_valid_file(self.file_path,
                                                      extension=['txt',
                                                                 '.png']),
                         self.file_path)

    def test_diff_extension(self):
        self.assertEqual(self.translator.is_valid_file(self.file_path,
                                                       extension=['jpeg',
                                                                  '.png']),
                         None)

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = self.translator.is_valid_file({'hello': 3}, extension=['jpeg',
                                                                      '.png'])
        with self.assertRaises(TypeError):
            _ = self.translator.is_valid_file('blah.txt', extension=[14,
                                                                     '.png'])

    def test_folder(self):
        self.assertEqual(self.translator.is_valid_file(os.path.abspath('.'),
                                                       extension='.txt'),
                         None)


class TestTranslateNotImplemented(unittest.TestCase):

    def test_empty_translator(self):
        if sys.version_info.major == 2:
            # Unable to assert TyperError when instantiating empty Translator
            return
            """
            with self.assertRaises(TypeError):
                _ = Translator()
                return
            """

        tran = Translator()

        with self.assertRaises(NotImplementedError):
            tran.translate('blah.txt')


if __name__ == '__main__':
    unittest.main()
