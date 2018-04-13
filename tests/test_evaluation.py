import unittest

import numpy as np

from evaluation.grade import WeightedSumGrade


class TestGrade(unittest.TestCase):

    def setUp(self):
        self.sample_fitness_vector = np.ones(3)
        self.sample_fitness_weigts = np.ones(3) + 1

    def test_weighted_sum_grader(self):
        grader = WeightedSumGrade(weights=self.sample_fitness_weigts)
        calced = grader(self.sample_fitness_vector)
        self.assertEqual(calced, self.sample_fitness_vector @ self.sample_fitness_weigts)
