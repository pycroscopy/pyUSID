# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import os
import numpy as np
import dask.array as da
import h5py
sys.path.append("../../pyUSID/")
from pyUSID.io import dtype_utils
from .data_utils import delete_existing_file

struc_dtype = np.dtype({'names': ['r', 'g', 'b'],
                        'formats': [np.float32, np.uint16, np.float64]})

file_path = 'test_dtype_utils.h5'


def compare_structured_arrays(arr_1, arr_2):
    """
    if not isinstance(arr_1, np.ndarray):
        raise TypeError("arr_1 was not a numpy array")
    if not isinstance(arr_2, np.ndarray):
        raise TypeError("arr_2 was not a numpy array")
    """
    if arr_1.dtype != arr_2.dtype:
        return False
    if arr_1.shape != arr_2.shape:
        return False
    tests = []
    for name in arr_1.dtype.names:
        tests.append(np.allclose(arr_1[name], arr_2[name]))
    return np.all(tests)


class TestDtypeUtils(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(file_path):
            with h5py.File(file_path, mode='w') as h5_f:
                num_elems = (5, 7)
                structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
                structured_array['r'] = 450 * np.random.random(size=num_elems)
                structured_array['g'] = np.random.randint(0, high=1024, size=num_elems)
                structured_array['b'] = 3178 * np.random.random(size=num_elems)
                _ = h5_f.create_dataset('compound', data=structured_array)
                _ = h5_f.create_dataset('real', data=450 * np.random.random(size=num_elems))
                _ = h5_f.create_dataset('real2', data=450 * np.random.random(size=(5, 7, 6)))
                _ = h5_f.create_dataset('complex', data=np.random.random(size=num_elems) +
                                                        1j * np.random.random(size=num_elems), dtype=np.complex64)
                h5_f.flush()
        return
    
    def tearDown(self):
        os.remove(file_path)


class TestContainsIntegers(unittest.TestCase):

    def test_typical(self):
        self.assertTrue(dtype_utils.contains_integers([1, 2, -3, 4]))
        self.assertTrue(dtype_utils.contains_integers(range(5)))
        self.assertTrue(dtype_utils.contains_integers([2, 5, 8, 3], min_val=2))
        self.assertTrue(dtype_utils.contains_integers(np.arange(5)))
        self.assertFalse(dtype_utils.contains_integers(np.arange(5), min_val=2))
        self.assertFalse(dtype_utils.contains_integers([1, 4.5, 2.2, -1]))
        self.assertFalse(dtype_utils.contains_integers([1, -2, 5], min_val=1))
        self.assertFalse(dtype_utils.contains_integers(['dsss', 34, 1.23, None]))
        self.assertFalse(dtype_utils.contains_integers([]))
        
        with self.assertRaises(TypeError):
            _ = dtype_utils.contains_integers(None)
        with self.assertRaises(TypeError):
            _ = dtype_utils.contains_integers(14)

    def test_illegal_min_val(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.contains_integers([1, 2, 3, 4], min_val='hello')

        with self.assertRaises(TypeError):
            _ = dtype_utils.contains_integers([1, 2, 3, 4], min_val=[1, 2])

        with self.assertRaises(ValueError):
            _ = dtype_utils.contains_integers([1, 2, 3, 4], min_val=1.234)


class TestIntegersToSlices(unittest.TestCase):

    def test_illegal_inputs(self):
        with self.assertRaises(TypeError):
            dtype_utils.integers_to_slices(slice(1, 15))
        with self.assertRaises(ValueError):
            dtype_utils.integers_to_slices([-1.43, 34.6565, 45.344, 5+6j])
        with self.assertRaises(ValueError):
            dtype_utils.integers_to_slices(['asdds', None, True, 45.344, 5 + 6j])

    def test_positive(self):
        expected = [slice(0, 3), slice(7, 8), slice(14, 18), slice(22, 23), slice(27, 28), slice(29, 30), slice(31, 32)]
        inputs = np.hstack([range(item.start, item.stop) for item in expected])
        ret_val = dtype_utils.integers_to_slices(inputs)
        self.assertEqual(expected, ret_val)

    def test_negative(self):
        expected = [slice(-7, -4), slice(-2, 3), slice(14, 18), slice(22, 23), slice(27, 28), slice(29, 30)]
        inputs = np.hstack([range(item.start, item.stop) for item in expected])
        ret_val = dtype_utils.integers_to_slices(inputs)
        self.assertEqual(expected, ret_val)


class TestStackRealToComplex(unittest.TestCase):

    def test_single(self):
        expected = 4.32 + 5.67j
        real_val = [np.real(expected), np.imag(expected)]
        actual = dtype_utils.stack_real_to_complex(real_val)
        self.assertTrue(np.allclose(actual, expected))

    def test_1d(self):
        expected = 5 * np.random.rand(6) + 7j * np.random.rand(6)
        real_val = np.hstack([np.real(expected), np.imag(expected)])
        actual = dtype_utils.stack_real_to_complex(real_val)
        self.assertTrue(np.allclose(actual, expected))

    def test_2d(self):
        expected = 5 * np.random.rand(2, 8) + 7j * np.random.rand(2, 8)
        real_val = np.hstack([np.real(expected), np.imag(expected)])
        actual = dtype_utils.stack_real_to_complex(real_val)
        self.assertTrue(np.allclose(actual, expected))

    def base_nd(self, lazy_in, lazy):
        expected = 5 * np.random.rand(2, 3, 5, 8) + 7j * np.random.rand(2, 3, 5, 8)
        real_val = np.concatenate([np.real(expected), np.imag(expected)], axis=3)
        if lazy_in:
            real_val = da.from_array(real_val, chunks=real_val.shape)
        actual = dtype_utils.stack_real_to_complex(real_val, lazy=lazy)
        if lazy:
            self.assertIsInstance(actual, da.core.Array)
            actual = actual.compute()
        self.assertTrue(np.allclose(actual, expected))

    def test_nd_numpy_not_lazy(self):
        self.base_nd(False, False)

    def test_nd_numpy_lazy(self):
        self.base_nd(False, True)

    def test_nd_dask(self):
        self.base_nd(True, True)

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.stack_real_to_complex("Hello")

        with self.assertRaises(TypeError):
            _ = dtype_utils.stack_real_to_complex([1, {'a': 1}, 'a', True])


class TestStackRealToComplexHDF5(TestDtypeUtils):

    def base_h5_legal(self, lazy):
        with h5py.File(file_path, mode='r') as h5_f:
            h5_real = h5_f['real2']
            expected = h5_real[:, :, :3] + 1j * h5_real[:, :, 3:]
            actual = dtype_utils.stack_real_to_complex(h5_real, lazy=lazy)
            if lazy:
                self.assertIsInstance(actual, da.core.Array)
                actual = actual.compute()
            else:
                self.assertIsInstance(actual, np.ndarray)
            self.assertTrue(np.allclose(actual, expected))

    def test_h5_as_numpy(self):
        self.base_h5_legal(False)

    def test_h5_as_dask(self):
        self.base_h5_legal(True)

    def test_h5_illegal(self):
        with h5py.File(file_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                _ = dtype_utils.stack_real_to_complex(h5_f['complex'])

            with self.assertRaises(TypeError):
                _ = dtype_utils.stack_real_to_complex(h5_f['compound'])

    def test_illegal_odd_last_dim(self):
        expected = 5 * np.random.rand(2, 3, 5, 7) + 7j * np.random.rand(2, 3, 5, 7)
        real_val = np.concatenate([np.real(expected), np.imag(expected)[..., :-1]], axis=3)
        with self.assertRaises(ValueError):
            _ = dtype_utils.stack_real_to_complex(real_val)


class TestFlattenComplexToReal(unittest.TestCase):

    def test_1d_array(self):
        complex_array = 5 * np.random.rand(5) + 7j * np.random.rand(5)
        actual = dtype_utils.flatten_complex_to_real(complex_array)
        expected = np.hstack([np.real(complex_array), np.imag(complex_array)])
        self.assertTrue(np.allclose(actual, expected))

    def test_2d_array(self):
        complex_array = 5 * np.random.rand(2, 3) + 7j * np.random.rand(2, 3)
        actual = dtype_utils.flatten_complex_to_real(complex_array)
        expected = np.hstack([np.real(complex_array), np.imag(complex_array)])
        self.assertTrue(np.allclose(actual, expected))

    def test_nd_array_numpy(self):
        complex_array = 5 * np.random.rand(2, 3, 5, 7) + 7j * np.random.rand(2, 3, 5, 7)
        actual = dtype_utils.flatten_complex_to_real(complex_array)
        expected = np.concatenate([np.real(complex_array), np.imag(complex_array)], axis=3)
        self.assertTrue(np.allclose(actual, expected))

    def test_nd_array_dask(self):
        complex_array = 5 * np.random.rand(2, 3, 5, 7) + 7j * np.random.rand(2, 3, 5, 7)
        da_comp = da.from_array(complex_array, chunks=complex_array.shape)
        actual = dtype_utils.flatten_complex_to_real(da_comp)
        self.assertIsInstance(actual, da.core.Array)
        actual = actual.compute()
        expected = np.concatenate([np.real(complex_array), np.imag(complex_array)], axis=3)
        self.assertTrue(np.allclose(actual, expected))

    def test_real_in(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.flatten_complex_to_real(np.arange(4))

    def test_compound_illegal(self):
        num_elems = 5
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = np.random.random(size=num_elems)
        structured_array['g'] = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = np.random.random(size=num_elems)
        with self.assertRaises(TypeError):
            _ = dtype_utils.flatten_complex_to_real(structured_array)


class TestFlattenComplexToRealHDF5(TestDtypeUtils):

    def base_h5_legal(self, lazy):
        with h5py.File(file_path, mode='r') as h5_f:
            h5_comp = h5_f['complex']
            actual = dtype_utils.flatten_complex_to_real(h5_comp, lazy=lazy)
            if lazy:
                self.assertIsInstance(actual, da.core.Array)
                actual = actual.compute()
            expected = np.concatenate([np.real(h5_comp[()]), np.imag(h5_comp[()])], axis=len(h5_comp.shape) - 1)
            self.assertTrue(np.allclose(actual, expected))

    def test_h5_legal_not_lazy(self):
        self.base_h5_legal(False)

    def test_h5_legal_lazy(self):
        self.base_h5_legal(True)

    def test_h5_illegal(self):
        with h5py.File(file_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                _ = dtype_utils.flatten_complex_to_real(h5_f)

    def test_h5_illegal_dtype(self):
        with h5py.File(file_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                _ = dtype_utils.flatten_complex_to_real(h5_f['real'])

            with self.assertRaises(TypeError):
                _ = dtype_utils.flatten_complex_to_real(h5_f['compound'])


class TestGetCompoundSubTypes(unittest.TestCase):

    def test_legal(self):
        self.assertEqual({'r': np.float32, 'g': np.uint16, 'b': np.float64},
                         dtype_utils.get_compound_sub_dtypes(struc_dtype))

    def test_illegal(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.get_compound_sub_dtypes(np.float16)

        with self.assertRaises(TypeError):
            _ = dtype_utils.get_compound_sub_dtypes(16)

        with self.assertRaises(TypeError):
            _ = dtype_utils.get_compound_sub_dtypes(np.arange(4))


class TestStackRealToCompound(unittest.TestCase):

    def test_single(self):
        num_elems = 1
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.stack_real_to_compound(real_val, struc_dtype)
        self.assertTrue(compare_structured_arrays(actual, structured_array[0]))

    def test_1d(self):
        num_elems = 5
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.stack_real_to_compound(real_val, struc_dtype)
        self.assertTrue(compare_structured_arrays(actual, structured_array))

    def test_1d_list(self):
        num_elems = 5
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.stack_real_to_compound(list(real_val), struc_dtype)
        self.assertTrue(compare_structured_arrays(actual, structured_array))

    def base_nd(self, in_lazy, out_lazy):
        num_elems = (2, 3, 5, 7)
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals), axis=len(num_elems) - 1)
        if in_lazy:
            real_val = da.from_array(real_val, chunks=real_val.shape)
        actual = dtype_utils.stack_real_to_compound(real_val, struc_dtype, lazy=out_lazy)
        if out_lazy:
            self.assertIsInstance(actual, da.core.Array)
            actual = actual.compute()
        self.assertTrue(compare_structured_arrays(actual, structured_array))

    def test_nd_dask_in(self):
        with self.assertRaises(NotImplementedError):
            self.base_nd(True, True)

    def test_nd_numpy_in_not_lazy(self):
        self.base_nd(False, False)

    def test_nd_numpy_in_lazy(self):
        with self.assertRaises(NotImplementedError):
            self.base_nd(False, True)

    def test_illegal(self):
        num_elems = (3, 5)
        r_vals = np.random.random(size=num_elems)

        with self.assertRaises(TypeError):
            _ = dtype_utils.stack_real_to_compound(r_vals, np.float32)

        with self.assertRaises(ValueError):
            _ = dtype_utils.stack_real_to_compound(r_vals, struc_dtype)

        with self.assertRaises(ValueError):
            _ = dtype_utils.stack_real_to_compound([1, 'a', {'a': 1}, False], struc_dtype)


class TestFlattenCompoundToReal(unittest.TestCase):

    def test_single(self):
        num_elems = 1
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        expected = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.flatten_compound_to_real(structured_array[0])
        self.assertTrue(np.allclose(actual, expected))

    def test_1d(self):
        num_elems = 5
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        expected = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.flatten_compound_to_real(structured_array)
        self.assertTrue(np.allclose(actual, expected))

    def base_nd(self, lazy_in, lazy):
        num_elems = (5, 7, 2, 3)
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        if lazy_in:
            structured_array = da.from_array(structured_array, chunks=structured_array.shape)
        expected = np.concatenate((r_vals, g_vals, b_vals), axis=len(num_elems) - 1)
        actual = dtype_utils.flatten_compound_to_real(structured_array, lazy=lazy)
        if lazy:
            self.assertIsInstance(actual, da.core.Array)
            actual = actual.compute()
        self.assertTrue(np.allclose(actual, expected))

    def test_numpy_nd_not_lazy(self):
        self.base_nd(False, False)

    def test_numpy_nd_lazy(self):
        self.base_nd(False, True)

    def test_dask_nd(self):
        self.base_nd(True, True)

    def test_illegal(self):
        num_elems = (2, 3)
        r_vals = np.random.random(size=num_elems)
        with self.assertRaises(TypeError):
            _ = dtype_utils.flatten_compound_to_real(r_vals)

        with self.assertRaises(TypeError):
            _ = dtype_utils.flatten_compound_to_real(14)


class TestFlattenCompoundToRealHDF5(TestDtypeUtils):

    def base_nd_h5_legal(self, lazy):
        with h5py.File(file_path, mode='r') as h5_f:
            h5_comp = h5_f['compound']
            actual = dtype_utils.flatten_compound_to_real(h5_comp, lazy=lazy)
            if lazy:
                self.assertIsInstance(actual, da.core.Array)
                actual = actual.compute()
            expected = np.concatenate([h5_comp['r'], h5_comp['g'], h5_comp['b']], axis=len(h5_comp.shape) - 1)
            self.assertTrue(np.allclose(actual, expected))

    def test_h5_legal_not_lazy(self):
        self.base_nd_h5_legal(False)

    def test_h5_legal_lazy(self):
        self.base_nd_h5_legal(True)

    def test_nd_h5_illegal(self):
        with h5py.File(file_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                _ = dtype_utils.flatten_compound_to_real(h5_f['real'])

            with self.assertRaises(TypeError):
                _ = dtype_utils.flatten_compound_to_real(h5_f['complex'])


class TestStackRealToCompoundHDF5(TestDtypeUtils):

    def base_nd_h5_legal(self, lazy):
        with h5py.File(file_path, mode='r') as h5_f:
            h5_real = h5_f['real2']
            structured_array = np.zeros(shape=list(h5_real.shape)[:-1] + [h5_real.shape[-1] // len(struc_dtype.names)],
                                        dtype=struc_dtype)
            for name_ind, name in enumerate(struc_dtype.names):
                i_start = name_ind * structured_array.shape[-1]
                i_end = (name_ind + 1) * structured_array.shape[-1]
                structured_array[name] = h5_real[..., i_start:i_end]
            actual = dtype_utils.stack_real_to_compound(h5_real, struc_dtype, lazy=lazy)
            if lazy:
                self.assertIsInstance(actual, da.core.Array)
                actual = actual.compute()
            self.assertTrue(compare_structured_arrays(actual, structured_array))

    def test_nd_h5(self):
        self.base_nd_h5_legal(False)

    def test_nd_h5_lazy(self):
        with self.assertRaises(NotImplementedError):
            self.base_nd_h5_legal(True)

    def test_nd_h5_illegal(self):
        with h5py.File(file_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                _ = dtype_utils.stack_real_to_compound(h5_f['compound'], struc_dtype)

            with self.assertRaises(TypeError):
                _ = dtype_utils.stack_real_to_compound(h5_f['complex'], struc_dtype)


class TestFlattenToRealNumpy(unittest.TestCase):

    def test_real_nd(self):
        real_array = 5 * np.random.rand(2, 3, 5, 7)
        actual = dtype_utils.flatten_to_real(real_array)
        self.assertTrue(np.allclose(actual, real_array))

    def test_complex_nd(self):
        complex_array = 5 * np.random.rand(2, 3, 5, 7) + 7j * np.random.rand(2, 3, 5, 7)
        actual = dtype_utils.flatten_to_real(complex_array)
        expected = np.concatenate([np.real(complex_array), np.imag(complex_array)], axis=3)
        self.assertTrue(np.allclose(actual, expected))

    def test_complex_single(self):
        complex_val = 4.32 + 5.67j
        expected = [np.real(complex_val), np.imag(complex_val)]
        actual = dtype_utils.flatten_to_real(complex_val)
        self.assertTrue(np.allclose(actual, expected))

    def test_compound_nd(self):
        num_elems = (5, 7, 2, 3)
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        expected = np.concatenate((r_vals, g_vals, b_vals), axis=len(num_elems) - 1)
        actual = dtype_utils.flatten_to_real(structured_array)
        self.assertTrue(np.allclose(actual, expected))

    def test_compound_single(self):
        num_elems = 1
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        expected = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.flatten_to_real(structured_array[0])
        self.assertTrue(np.allclose(actual, expected))


class TestStackRealToTargetDtypeNumpy(unittest.TestCase):

    def test_complex_single(self):
        expected = 4.32 + 5.67j
        real_val = [np.real(expected), np.imag(expected)]
        actual = dtype_utils.stack_real_to_target_dtype(real_val, np.complex)
        self.assertTrue(np.allclose(actual, expected))

    def test_complex_nd(self):
        expected = 5 * np.random.rand(2, 3, 5, 8) + 7j * np.random.rand(2, 3, 5, 8)
        real_val = np.concatenate([np.real(expected), np.imag(expected)], axis=3)
        actual = dtype_utils.stack_real_to_target_dtype(real_val, np.complex)
        self.assertTrue(np.allclose(actual, expected))

    def test_compound_single(self):
        num_elems = 1
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.stack_real_to_target_dtype(real_val, struc_dtype)
        self.assertTrue(compare_structured_arrays(actual, structured_array[0]))

    def test_compound_nd(self):
        num_elems = (2, 3, 5, 7)
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals), axis=len(num_elems) - 1)
        actual = dtype_utils.stack_real_to_target_dtype(real_val, struc_dtype)
        self.assertTrue(compare_structured_arrays(actual, structured_array))

    def test_compound_nd(self):
        num_elems = (2, 3, 5, 7)
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024,
                                                           size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals),
                                  axis=len(num_elems) - 1)
        actual = dtype_utils.stack_real_to_target_dtype(real_val, struc_dtype)
        self.assertTrue(compare_structured_arrays(actual, structured_array))

    def test_cast_std_type(self):
        uint_array = np.random.randint(0, high=15, size=(3, 4, 5),
                                       dtype=np.uint16)
        actual = dtype_utils.stack_real_to_target_dtype(uint_array, np.float32)
        self.assertTrue(np.allclose(actual, np.float32(uint_array)))

    def test_cast_str_type(self):
        uint_array = np.random.randint(0, high=15, size=(3, 4, 5),
                                       dtype=np.uint16)
        actual = dtype_utils.stack_real_to_target_dtype(uint_array, np.dtype('<f4'))
        self.assertTrue(np.allclose(actual, np.float32(uint_array)))


class TestStackRealToTargetDtypeHDF5(TestDtypeUtils):

    def test_stack_real_nd_to_target_complex_h5_legal(self):
        with h5py.File(file_path, mode='r') as h5_f:
            h5_real = h5_f['real2']
            expected = h5_real[:, :, :3] + 1j * h5_real[:, :, 3:]
            actual = dtype_utils.stack_real_to_target_dtype(h5_real,
                                                            np.complex)
            self.assertTrue(np.allclose(actual, expected))


class TestCheckDtype(TestDtypeUtils):

    def test_real_numpy(self):
        # real_matrix = np.random.rand(5, 7)
        # func, is_complex, is_compound, n_features, type_mult = dtype_utils.check_dtype(real_matrix)
        with h5py.File(file_path, mode='r') as h5_f:
            func, is_complex, is_compound, n_features, type_mult = dtype_utils.check_dtype(h5_f['real'])
            self.assertEqual(func, h5_f['real'].dtype.type)
            self.assertEqual(is_complex, False)
            self.assertEqual(is_compound, False)
            self.assertEqual(n_features, h5_f['real'].shape[1])
            self.assertEqual(type_mult, h5_f['real'].dtype.type(0).itemsize)

    def test_complex_numpy(self):
        with h5py.File(file_path, mode='r') as h5_f:
            func, is_complex, is_compound, n_features, type_mult = dtype_utils.check_dtype(h5_f['complex'])
            self.assertEqual(func, dtype_utils.flatten_complex_to_real)
            self.assertEqual(is_complex, True)
            self.assertEqual(is_compound, False)
            self.assertEqual(n_features, 2 * h5_f['complex'].shape[1])
            self.assertEqual(type_mult, 2 * np.real(h5_f['complex'][0, 0]).dtype.itemsize)

    def test_compound_numpy(self):
        with h5py.File(file_path, mode='r') as h5_f:
            func, is_complex, is_compound, n_features, type_mult = dtype_utils.check_dtype(h5_f['compound'])
            self.assertEqual(func, dtype_utils.flatten_compound_to_real)
            self.assertEqual(is_complex, False)
            self.assertEqual(is_compound, True)
            self.assertEqual(n_features, 3 * h5_f['compound'].shape[1])
            self.assertEqual(type_mult, 3 * np.float32(0).itemsize)

    def test_non_hdf(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.check_dtype(np.arange(15))


class TestGetCompoundSubTypes(unittest.TestCase):

    def test_valid(self):
        input_dict = {'names': ['r', 'g', 'b'], 'formats': [np.float32, np.uint16, np.float64]}
        expected = dict()
        for name, dtype in zip(input_dict['names'], input_dict['formats']):
            expected[name] = dtype
        struct_dtype = np.dtype(input_dict)
        actual = dtype_utils.get_compound_sub_dtypes(struct_dtype)
        self.assertEqual(expected, actual)

    def test_invalid(self):
        for item in [np.float16, np.complex, 4, 2.343, True, 'ssdsds', np.arange(5)]:
            with self.assertRaises(TypeError):
                _ = dtype_utils.get_compound_sub_dtypes(item)


class TestValidateDtype(unittest.TestCase):

    def test_valid(self):
        struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
                                'formats': [np.float32, np.uint16, np.float64]})
        for dtype in [np.float32, np.float16, np.complex, np.complex64, np.uint8, np.int16, struct_dtype]:
            self.assertTrue(dtype_utils.validate_dtype(dtype))

    def test_invalid(self):
        for dtype in [6, 'dssds', np.arange(5), True]:
            with self.assertRaises(TypeError):
                dtype_utils.validate_dtype(dtype)


class TestIsComplexDtype(unittest.TestCase):

    def test_valid(self):
        struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
                                 'formats': [np.float32, np.uint16, np.float64]})
        for dtype in [np.float32, np.float16, np.uint8, np.int16, struct_dtype, bool]:
            self.assertFalse(dtype_utils.is_complex_dtype(dtype))

        for dtype in [np.complex, np.complex64, np.complex128]:
            self.assertTrue(dtype_utils.is_complex_dtype(dtype))


class TestGetExponent(unittest.TestCase):

    def test_negative_small(self):
        expected = -7
        self.assertEqual(expected,
                         dtype_utils.get_exponent(np.arange(5) * -10**expected))

    def test_positive_large(self):
        expected = 4
        self.assertEqual(expected,
                         dtype_utils.get_exponent(np.arange(6) * 10 ** expected))

    def test_mixed_large(self):
        expected = 4
        self.assertEqual(expected,
                         dtype_utils.get_exponent(np.random.randint(-8, high=3, size=(5, 5)) * 10 ** expected))

    def test_illegal_type(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.get_exponent('hello')
            _ = dtype_utils.get_exponent([1, 2, 3])


class TestValidateStringArgs(unittest.TestCase):

    def test_empty(self):
        with self.assertRaises(ValueError):
            _ = dtype_utils.validate_string_args(['      '], ['meh'])

    def test_spaces(self):
        expected = 'fd'
        [ret] = dtype_utils.validate_string_args(['  ' + expected + '    '], ['meh'])
        self.assertEqual(expected, ret)

    def test_single(self):
        expected = 'fd'
        [ret] = dtype_utils.validate_string_args(expected , 'meh')
        self.assertEqual(expected, ret)

    def test_multi(self):
        expected = ['abc', 'def']
        returned = dtype_utils.validate_string_args(['   ' + expected[0], expected[1] + '    '], ['meh', 'foo'])
        for exp, ret in zip(expected, returned):
            self.assertEqual(exp, ret)

    def test_not_string_lists(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.validate_string_args([14], ['meh'])

        with self.assertRaises(TypeError):
            _ = dtype_utils.validate_string_args(14, ['meh'])

        with self.assertRaises(TypeError):
            _ = dtype_utils.validate_string_args({'dfdf': 14}, ['meh'])

    def test_name_not_string(self):
        actual = ['ghghg']
        ret = dtype_utils.validate_string_args(actual, [np.arange(3)])
        self.assertEqual(ret, actual)

    def test_unequal_lengths(self):
        expected = ['a', 'b']
        actual = dtype_utils.validate_string_args(expected + ['c'], ['a', 'b'])
        for exp, ret in zip(expected, actual):
            self.assertEqual(exp, ret)

    def test_names_not_list_of_strings(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.validate_string_args(['a', 'v'], {'a': 1, 'v': 43})


class TestLazyLoadArray(unittest.TestCase):

    def test_dask_input(self):
        arr = da.random.random(2, 3)
        self.assertTrue(np.allclose(arr, dtype_utils.lazy_load_array(arr)))

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.lazy_load_array([1, 2, 3])

    def test_numpy(self):
        np_arr = np.random.rand(2, 3)
        da_arr = da.from_array(np_arr, chunks=np_arr.shape)
        self.assertTrue(np.allclose(da_arr, dtype_utils.lazy_load_array(np_arr)))

    def test_h5_dset_no_chunks(self):
        h5_path = 'blah.h5'
        delete_existing_file(h5_path)
        with h5py.File(h5_path, mode='w') as h5_f:
            np_arr = np.random.rand(2, 3)
            h5_dset = h5_f.create_dataset('Test', data=np_arr)
            da_arr = da.from_array(np_arr, chunks=np_arr.shape)
            self.assertTrue(np.allclose(da_arr, dtype_utils.lazy_load_array(h5_dset)))
        os.remove(h5_path)

    def test_h5_dset_w_chunks(self):
        h5_path = 'blah.h5'
        delete_existing_file(h5_path)
        with h5py.File(h5_path, mode='w') as h5_f:
            np_arr = np.random.rand(200, 30)
            h5_dset = h5_f.create_dataset('Test', data=np_arr, chunks=(1, 30))
            da_arr = da.from_array(np_arr, chunks=h5_dset.chunks)
            self.assertTrue(np.allclose(da_arr, dtype_utils.lazy_load_array(h5_dset)))
        os.remove(h5_path)


if __name__ == '__main__':
    unittest.main()
