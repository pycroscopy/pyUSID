# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
from multiprocessing import cpu_count

sys.path.append("../../pyUSID/")
from pyUSID.processing import comp_utils

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

        self.assertEqual(mem, comp_utils.get_available_memory())


if __name__ == '__main__':
    unittest.main()