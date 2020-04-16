# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import numpy as np
from multiprocessing import cpu_count

sys.path.append("../../pyUSID/")
from pyUSID.processing import comp_utils
from ..io import data_utils

MAX_CPU_CORES = cpu_count()


class TestRecommendCPUCores(unittest.TestCase):

    def test_many_small_jobs(self):
        num_jobs = 14035
        if MAX_CPU_CORES > 4:
            min_free_cores = 2
        else:
            min_free_cores = 1
        ret_val = comp_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False)
        self.assertEqual(ret_val, max(1, MAX_CPU_CORES-min_free_cores))
        ret_val = comp_utils.recommend_cpu_cores(num_jobs, requested_cores=1, lengthy_computation=False)
        self.assertEqual(ret_val, 1)
        ret_val = comp_utils.recommend_cpu_cores(num_jobs, requested_cores=MAX_CPU_CORES, lengthy_computation=False)
        self.assertEqual(ret_val, MAX_CPU_CORES)
        ret_val = comp_utils.recommend_cpu_cores(num_jobs, requested_cores=5000, lengthy_computation=False)
        self.assertEqual(ret_val, MAX_CPU_CORES)

    def test_changing_min_cores(self):
        num_jobs = 14035
        for min_free_cores in range(1, MAX_CPU_CORES):
            ret_val = comp_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False, min_free_cores=min_free_cores)
            self.assertEqual(ret_val, max(1, MAX_CPU_CORES - min_free_cores))

    def test_illegal_min_free_cores(self):
        num_jobs = 14035
        min_free_cores = MAX_CPU_CORES
        with self.assertRaises(ValueError):
            _ = comp_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False, min_free_cores=min_free_cores)

    def test_few_small_jobs(self):
        num_jobs = 13
        ret_val = comp_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False)
        self.assertEqual(ret_val, 1)
        ret_val = comp_utils.recommend_cpu_cores(num_jobs, requested_cores=MAX_CPU_CORES, lengthy_computation=False)
        self.assertEqual(ret_val, 1)

    def test_few_large_jobs(self):
        num_jobs = 13
        if MAX_CPU_CORES > 4:
            min_free_cores = 2
        else:
            min_free_cores = 1
        ret_val = comp_utils.recommend_cpu_cores(num_jobs, lengthy_computation=True)
        self.assertEqual(ret_val, max(1, MAX_CPU_CORES-min_free_cores))
        ret_val = comp_utils.recommend_cpu_cores(num_jobs, requested_cores=MAX_CPU_CORES - 1, lengthy_computation=True)
        self.assertEqual(ret_val, max(1, MAX_CPU_CORES - 1))

    def test_invalid_min_cores(self):
        with self.assertRaises(TypeError):
            _ = comp_utils.recommend_cpu_cores(14035, min_free_cores=[4])

    def test_invalid_requested_cores(self):
        with self.assertRaises(TypeError):
            _ = comp_utils.recommend_cpu_cores(14035, requested_cores=[4])

    def test_invalid_num_jobs(self):
        with self.assertRaises(TypeError):
            _ = comp_utils.recommend_cpu_cores(14035.67)

        with self.assertRaises(ValueError):
            _ = comp_utils.recommend_cpu_cores(-14035)

        with self.assertRaises(TypeError):
            _ = comp_utils.recommend_cpu_cores('not a number')


class TestGetAvailableMemory(unittest.TestCase):

    def test_standard(self):
        from psutil import virtual_memory
        mem = virtual_memory().available

        if sys.maxsize <= 2 ** 32:
            mem = min([mem, sys.maxsize])

        self.assertTrue(abs(mem-comp_utils.get_available_memory()) < 0.01 * virtual_memory().total)


class TestGetMPI(unittest.TestCase):

    def test_standard(self):
        try:
            from mpi4py import MPI
            if MPI.COMM_WORLD.Get_size() == 1:
                # mpi4py available but NOT called via mpirun
                MPI = None
        except ImportError:
            # mpi4py not even present! Single node by default:
            MPI = None

        self.assertAlmostEqual(comp_utils.get_MPI(), MPI)


def func_w_args_only(vec, arg_1, arg_2):
    return (vec - arg_1) * arg_2


def func_w_kwargs_only(vec, arg_1=0, arg_2=1):
    return(vec - arg_1) * arg_2


def func_w_args_and_kwargs(vec, arg_1, arg_2=1):
    return (vec - arg_1) * arg_2


class TestParallelCompute(unittest.TestCase):

    def test_invalid_func_type(self):
        with self.assertRaises(TypeError):
            _ = comp_utils.parallel_compute(np.random.rand(2, 5),
                                            'not callable')

        with self.assertRaises(TypeError):
            _ = comp_utils.parallel_compute(np.random.rand(2, 5),
                                            {'a': 1})

    def test_invalid_data_type(self):
        with self.assertRaises(TypeError):
            _ = comp_utils.parallel_compute([[1, 2, 3], [4, 5, 6]],
                                            np.mean)

    def test_invalid_func_args_type(self):
        with self.assertRaises(TypeError):
            _ = comp_utils.parallel_compute(np.random.rand(2, 5), np.mean,
                                            func_args={'should be': 'a list'})

    def test_invalid_func_kwargs_type(self):
        with self.assertRaises(TypeError):
            _ = comp_utils.parallel_compute(np.random.rand(2, 5), np.mean,
                                            func_kwargs=['not', 'a', 'dict'])

    def base_parallel_compute(self, num_job_scaler=50, cores=None,
                              lengthy_computation=False, func_args=None,
                              expected_cores=None, func_kwargs=None):
        data = np.random.rand(MAX_CPU_CORES * num_job_scaler, 5)
        expected = np.mean(data, axis=1)
        if expected_cores is None:
            expected_cores = MAX_CPU_CORES - 1 - int(MAX_CPU_CORES > 4)

        exp_log = 'Rank 0 starting computing on {} cores (requested {} cores' \
                  ')'.format(expected_cores, cores)
        with data_utils.capture_stdout() as get_stdout:
            result = comp_utils.parallel_compute(data, np.mean,
                                                 cores=cores,
                                                 lengthy_computation=lengthy_computation,
                                                 func_args=func_args,
                                                 func_kwargs=func_kwargs,
                                                 verbose=True)
            logs = get_stdout()
        self.assertIsInstance(result, list)
        self.assertEqual(data.shape[0], len(result))
        self.assertTrue(np.allclose(expected, np.array(result)))
        print(logs)
        self.assertTrue(exp_log in logs)

    def test_standard_joblib_compute_many_jobs(self):
        self.base_parallel_compute(num_job_scaler=50)

    def test_standard_serial_compute_few_jobs(self):
        self.base_parallel_compute(num_job_scaler=5, expected_cores=1)

    def test_standard_joblib_compute_few_long_jobs(self):
        self.base_parallel_compute(num_job_scaler=5, lengthy_computation=True)

    def test_respect_requested_cores(self):
        self.base_parallel_compute(num_job_scaler=30, cores=2,
                                   expected_cores=2)

    def base_with_custom_func(self, use_args=False, use_kwargs=False,
                              cores=None, expected_cores=None):

        if not use_args and not use_kwargs:
            use_args = True

        func_args = None
        func_kwargs = None

        if use_args and not use_kwargs:
            func = func_w_args_only
            func_args = [0.5, 3]
        elif not use_args and use_kwargs:
            func = func_w_kwargs_only
            func_kwargs = {'arg_1': 0.5, 'arg_2': 3}
        elif use_args and use_kwargs:
            func = func_w_args_and_kwargs
            func_args = [0.5]
            func_kwargs = {'arg_2': 3}

        print('func: {}, args: {}, kwargs: {}'.format(func, func_args,
                                                      func_kwargs))

        data = np.random.rand(MAX_CPU_CORES * 50, 5)
        expected = (data - 0.5) * 3
        if expected_cores is None:
            expected_cores = MAX_CPU_CORES - 1 - int(MAX_CPU_CORES > 4)

        exp_log = 'Rank 0 starting computing on {} cores (requested {} cores' \
                  ')'.format(expected_cores, cores)
        with data_utils.capture_stdout() as get_stdout:
            result = comp_utils.parallel_compute(data, func,
                                                 cores=cores,
                                                 lengthy_computation=False,
                                                 func_args=func_args,
                                                 func_kwargs=func_kwargs,
                                                 verbose=True)
            logs = get_stdout()
        self.assertIsInstance(result, list)
        self.assertEqual(data.shape[0], len(result))
        self.assertTrue(np.allclose(expected, np.array(result)))
        print(logs)
        self.assertTrue(exp_log in logs)

    def test_func_args_only_parallel(self):
        self.base_with_custom_func(use_args=True, use_kwargs=False,
                                   cores=None, expected_cores=None)

    def test_func_kwargs_only_parallel(self):
        self.base_with_custom_func(use_args=False, use_kwargs=True,
                                   cores=None, expected_cores=None)

    def test_func_args_and_kwargs_parallel(self):
        self.base_with_custom_func(use_args=True, use_kwargs=True,
                                   cores=None, expected_cores=None)

    def test_func_args_only_serial(self):
        self.base_with_custom_func(use_args=True, use_kwargs=False,
                                   cores=1, expected_cores=1)

    def test_func_kwargs_only_serial(self):
        self.base_with_custom_func(use_args=False, use_kwargs=True,
                                   cores=1, expected_cores=1)

    def test_func_args_and_kwargs_serial(self):
        self.base_with_custom_func(use_args=True, use_kwargs=True,
                                   cores=1, expected_cores=1)


if __name__ == '__main__':
    unittest.main()
