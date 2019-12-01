# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import numpy as np

sys.path.append("../../pyUSID/")
from pyUSID.io import write_utils

if sys.version_info.major == 3:
    unicode = str


class TestMakeIndicesMatrix(unittest.TestCase):

    def test_dim_w_val_1(self):
        with self.assertRaises(ValueError):
            _ = write_utils.make_indices_matrix([1, 2, 3])

    def test_non_int_dim_sizes(self):
        with self.assertRaises(ValueError):
            _ = write_utils.make_indices_matrix([1.233, 2.4, 3])

    def test_not_list(self):
        with self.assertRaises(TypeError):
            _ = write_utils.make_indices_matrix(1)

    def test_weird_inputs(self):
        with self.assertRaises(ValueError):
            _ = write_utils.make_indices_matrix([2, 'hello', 3])

    def test_matrix_1_dims(self):
        expected = np.expand_dims(np.arange(4), axis=0)
        ret_val = write_utils.make_indices_matrix([4], is_position=False)
        self.assertTrue(np.allclose(expected, ret_val))
        ret_val = write_utils.make_indices_matrix([4], is_position=True)
        self.assertTrue(np.allclose(expected.T, ret_val))

    def test_2_dims(self):
        expected = np.vstack((np.tile(np.arange(2), 3),
                              np.repeat(np.arange(3), 2)))
        ret_val = write_utils.make_indices_matrix([2, 3], is_position=False)
        self.assertTrue(np.allclose(expected, ret_val))
        ret_val = write_utils.make_indices_matrix([2, 3], is_position=True)
        self.assertTrue(np.allclose(expected.T, ret_val))

    def test_3_dims(self):
        expected = np.vstack((np.tile(np.arange(2), 3 * 4),
                              np.tile(np.repeat(np.arange(3), 2), 4),
                              np.repeat(np.arange(4), 6)))
        ret_val = write_utils.make_indices_matrix([2, 3, 4], is_position=False)
        self.assertTrue(np.allclose(expected, ret_val))
        ret_val = write_utils.make_indices_matrix([2, 3, 4], is_position=True)
        self.assertTrue(np.allclose(expected.T, ret_val))


class TestGetAuxDsetSlicing(unittest.TestCase):

    def test_legal_single_dim(self):
        ret_val = write_utils.get_aux_dset_slicing(['X'], is_spectroscopic=True)
        expected = {'X': (slice(0, 1), slice(None))}
        self.assertEqual(ret_val, expected)

        ret_val = write_utils.get_aux_dset_slicing(['X'], is_spectroscopic=False)
        expected = {'X': (slice(None), slice(0, 1))}
        self.assertEqual(ret_val, expected)

    def test_legal_multi_dim(self):
        ret_val = write_utils.get_aux_dset_slicing(['X', 'Y'], is_spectroscopic=True)
        expected = {'X': (slice(0, 1), slice(None)), 'Y': (slice(1, 2), slice(None))}
        self.assertEqual(ret_val, expected)

        ret_val = write_utils.get_aux_dset_slicing(['X', 'Y'], is_spectroscopic=False)
        expected = {'X': (slice(None), slice(0, 1)), 'Y': (slice(None), slice(1, 2))}
        self.assertEqual(ret_val, expected)

    def test_odd_input(self):
        with self.assertRaises(TypeError):
            _ = write_utils.get_aux_dset_slicing([1, 'Y'], is_spectroscopic=True)
        with self.assertRaises(ValueError):
            _ = write_utils.get_aux_dset_slicing([], is_spectroscopic=True)


class TestCleanStringAtt(unittest.TestCase):
            
    def test_float(self):
        expected = 5.321
        self.assertEqual(expected, write_utils.clean_string_att(expected))

    def test_str(self):
        expected = 'test'
        self.assertEqual(expected, write_utils.clean_string_att(expected))

    def test_num_array(self):
        expected = [1, 2, 3.456]
        self.assertEqual(expected, write_utils.clean_string_att(expected))

    def test_str_list(self):
        expected = ['a', 'bc', 'def']
        returned = write_utils.clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)

    def test_str_tuple(self):
        expected = ('a', 'bc', 'def')
        returned = write_utils.clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)


class TestDimension(unittest.TestCase):

    def test_values_as_array(self):
        name = 'Bias'
        units = 'V'
        values = np.random.rand(5)

        descriptor = write_utils.Dimension(name, units, values)
        for expected, actual in zip([name, units, values],
                                    [descriptor.name, descriptor.units, descriptor.values]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

    def test_values_as_length(self):
        name = 'Bias'
        units = 'V'
        values = np.arange(5)

        descriptor = write_utils.Dimension(name, units, len(values))
        for expected, actual in zip([name, units],
                                    [descriptor.name, descriptor.units]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))
        self.assertTrue(np.allclose(values, descriptor.values))

    def test_repr(self):
        name = 'Bias'
        units = 'V'
        values = np.arange(5)

        descriptor = write_utils.Dimension(name, units, len(values))
        actual = '{}'.format(descriptor)
        expected = '{} ({}) mode:{} : {}'.format(name, units, descriptor.mode, values)
        self.assertEqual(actual, expected)

    def test_equality(self):
        name = 'Bias'
        units = 'V'

        dim_1 = write_utils.Dimension(name, units, [0, 1, 2, 3, 4])
        dim_2 = write_utils.Dimension(name, units, np.arange(5, dtype=np.float32))
        self.assertEqual(dim_1, dim_2)

    def test_inequality(self):
        name = 'Bias'
        units = 'V'

        self.assertNotEqual(write_utils.Dimension(name, units, [0, 1, 2, 3]),
                            write_utils.Dimension(name, units, [0, 1, 2, 4]))

        self.assertNotEqual(write_utils.Dimension('fdfd', units, [0, 1, 2, 3]),
                            write_utils.Dimension(name, units, [0, 1, 2, 3]))

        self.assertNotEqual(write_utils.Dimension(name, 'fdfd', [0, 1, 2, 3]),
                            write_utils.Dimension(name, units, [0, 1, 2, 3]))

        self.assertNotEqual(write_utils.Dimension(name, units, [0, 1, 2, 3],
                                                  mode=write_utils.DimType.DEPENDENT),
                            write_utils.Dimension(name, units, [0, 1, 2, 3],
                                                  mode=write_utils.DimType.INCOMPLETE))

        self.assertNotEqual(write_utils.Dimension(name, units, [0, 1, 2]),
                            write_utils.Dimension(name, units, [0, 1, 2, 3]))

    def test_invalid_mode(self):
        with self.assertRaises(TypeError):
            _ = write_utils.Dimension('Name', 'units', 5, mode='Incomplete')

    def test_default_mode(self):
        dim = write_utils.Dimension('Name', 'units', 1)
        self.assertEqual(dim.mode, write_utils.DimType.DEFAULT)

    def test_illegal_instantiation(self):

        with self.assertRaises(TypeError):
            _ = write_utils.Dimension('Name', 14, np.arange(4))

        with self.assertRaises(TypeError):
            _ = write_utils.Dimension(14, 'nm', np.arange(4))

        with self.assertRaises(ValueError):
            _ = write_utils.Dimension('Name', 'unit', 0)

        with self.assertRaises(TypeError):
            _ = write_utils.Dimension('Name', 'unit', 'invalid')


class TestDimType(unittest.TestCase):

    def test_dim_type_invalid_comparison(self):
        with self.assertRaises(TypeError):
            write_utils.DimType.INCOMPLETE == "Default"

    def test_dim_type_valid_comparison(self):
        self.assertTrue(write_utils.DimType.DEFAULT < write_utils.DimType.INCOMPLETE)
        self.assertTrue(write_utils.DimType.INCOMPLETE < write_utils.DimType.DEPENDENT)


class TestBuildIndValMatrices(unittest.TestCase):

    def test_empty(self):
        inds, vals = write_utils.build_ind_val_matrices([[0]], is_spectral=True)
        self.assertTrue(np.allclose(inds, write_utils.INDICES_DTYPE(np.expand_dims(np.arange(1), 0))))
        self.assertTrue(np.allclose(vals, write_utils.VALUES_DTYPE(np.expand_dims(np.arange(1), 0))))

    def test_1D(self):
        sine_val = np.sin(np.linspace(0, 2*np.pi, 128))
        inds, vals = write_utils.build_ind_val_matrices([sine_val], is_spectral=True)
        self.assertTrue(np.allclose(inds, write_utils.INDICES_DTYPE(np.expand_dims(np.arange(len(sine_val)), axis=0))))
        self.assertTrue(np.allclose(vals, write_utils.VALUES_DTYPE(np.expand_dims(sine_val, axis=0))))

    def test_1D_pos(self):
        sine_val = np.sin(np.linspace(0, 2 * np.pi, 128))
        inds, vals = write_utils.build_ind_val_matrices([sine_val], is_spectral=False)
        self.assertTrue(np.allclose(inds, write_utils.INDICES_DTYPE(np.expand_dims(np.arange(len(sine_val)), axis=1))))
        self.assertTrue(np.allclose(vals, write_utils.VALUES_DTYPE(np.expand_dims(sine_val, axis=1))))

    def test_3D(self):
        max_v = 4
        half_pts = 8
        bi_triang = np.roll(np.hstack((np.linspace(-max_v, max_v, half_pts, endpoint=False),
                                       np.linspace(max_v, -max_v, half_pts, endpoint=False))), -half_pts // 2)
        cycles = [0, 1, 2]
        fields = [0, 1]
        exp_vals = np.vstack((np.tile(bi_triang, 6), np.tile(np.repeat(fields, 2 * half_pts), 3),
                              np.repeat(cycles, 2 * 2 * half_pts)))
        exp_inds = np.vstack((np.tile(np.arange(2 * half_pts), 6), np.tile(np.repeat(fields, 2 * half_pts), 3),
                              np.repeat(cycles, 2 * 2 * half_pts)))
        inds, vals = write_utils.build_ind_val_matrices([bi_triang, fields, cycles])
        self.assertTrue(np.allclose(exp_inds, inds))
        self.assertTrue(np.allclose(exp_vals, vals))

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            _ = write_utils.build_ind_val_matrices("not a list of arrays")

        with self.assertRaises(ValueError):
            _ = write_utils.build_ind_val_matrices([[0, 1], np.random.randint(0, high=5, size=(3, 4))])


class TestCreateSpecIndsFromVals(unittest.TestCase):

    def test_legal(self):
        max_v = 4
        half_pts = 8
        bi_triang = np.roll(np.hstack((np.linspace(-max_v, max_v, half_pts, endpoint=False),
                                       np.linspace(max_v, -max_v, half_pts, endpoint=False))), -half_pts // 2)
        cycles = [0, 1, 2]
        fields = [0, 1]
        exp_vals = np.vstack((np.tile(bi_triang, 6), np.tile(np.repeat(fields, 2 * half_pts), 3),
                              np.repeat(cycles, 2 * 2 * half_pts)))
        exp_inds = np.vstack((np.tile(np.arange(2 * half_pts), 6), np.tile(np.repeat(fields, 2 * half_pts), 3),
                              np.repeat(cycles, 2 * 2 * half_pts)))

        inds = write_utils.create_spec_inds_from_vals(exp_vals)
        self.assertTrue(np.allclose(inds, exp_inds))

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            _ = write_utils.create_spec_inds_from_vals([[0, 1, 0, 1],
                                                        [0, 0, 1, 1]])

        with self.assertRaises(ValueError):
            _ = write_utils.create_spec_inds_from_vals(np.random.rand(2, 3, 4))


class TestCalcChunks(unittest.TestCase):

    def test_no_unit_chunk(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = None
        ret_val = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)
        self.assertTrue(np.allclose(ret_val, (26, 100)))

    def test_unit_chunk(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = (3, 7)
        ret_val = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)
        self.assertTrue(np.allclose(ret_val, (27, 98)))

    def test_no_unit_chunk_max_mem(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = None
        max_mem = 50000
        ret_val = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks, max_chunk_mem=max_mem)
        self.assertTrue(np.allclose(ret_val, (56, 224)))

    def test_unit_chunk_max_mem(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = (3, 7)
        max_mem = 50000
        ret_val = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks, max_chunk_mem=max_mem)
        self.assertTrue(np.allclose(ret_val, (57, 224)))

    def test_unit_not_iterable(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = 4

        with self.assertRaises(TypeError):
            _ = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)

    def test_shape_mismatch(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = (1, 5, 9)

        with self.assertRaises(ValueError):
            _ = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)

    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            _ = write_utils.calc_chunks("Fdfd", 14)

        with self.assertRaises(TypeError):
            _ = write_utils.calc_chunks((16384, 16384 * 4), 2.124)


class TestGetSlope(unittest.TestCase):

    def test_linear(self):
        expected = 0.25
        actual = write_utils.get_slope(np.arange(-1, 1, expected))
        self.assertEqual(expected, actual)

    def test_linear_dirty(self):
        # When reading from HDF5, rounding errors can result in minor variations in the diff
        expected = 0.25E-9
        vector = np.arange(-1E-9, 1E-9, expected)
        round_error = np.random.rand(vector.size) * 1E-14
        vector += round_error
        actual = write_utils.get_slope(vector, tol=1E-3)
        self.assertAlmostEqual(expected, actual)

    def test_invalid_tolerance(self):
        with self.assertRaises(TypeError):
            _ = write_utils.get_slope(np.sin(np.arange(4)), tol="hello")

    def test_non_linear(self):
        with self.assertRaises(ValueError):
            _ = write_utils.get_slope(np.sin(np.arange(4)))

    def test_invalid_inputs(self):
        with self.assertRaises(BaseException):
             _ = write_utils.get_slope("hello")


class TestToRanges(unittest.TestCase):

    def test_valid(self):
        actual = write_utils.to_ranges([0, 1, 2, 3, 7, 8, 9, 10])
        actual = list(actual)
        if sys.version_info.major == 3:
            expected = [range(0, 4), range(7, 11)]
            self.assertTrue(all([x == y for x, y in zip(expected, actual)]))
        else:
            expected = [xrange(0, 4), xrange(7, 11)]
            for in_x, out_x in zip(expected, actual):
                self.assertTrue(all([x == y for x, y in zip(list(in_x), list(out_x))]))


if __name__ == '__main__':
    unittest.main()
