import unittest

import numpy as np

from evolute.operators import SmoothMate, RandomPickMate, DefaultMate
from evolute.operators import UniformMutation
from evolute.operators import ElitismSelection


class TestMate(unittest.TestCase):

    def setUp(self):
        self.sample_individuals = [
            np.zeros(3), np.ones(3) + 1
        ]

    def test_random_pick_mated_offspring_only_contains_entries_from_parents(self):
        mater = RandomPickMate()
        offspring = mater(*self.sample_individuals)
        eq = np.logical_or(offspring == 0., offspring == 2.)
        self.assertTrue(np.all(eq))

    def test_smooth_mate_produces_offspring_which_is_the_mean_of_parents(self):
        mater = SmoothMate()
        offspring = mater(*self.sample_individuals)
        self.assertTrue(np.all(offspring == np.ones_like(offspring)))


class TestMutate(unittest.TestCase):

    def setUp(self):
        self.sample_individuals = np.zeros((3, 4))

    def test_uniform_locuswise_op_mutates_every_locus_with_rate_1(self):
        # Test may fail in the very unlikely case of a mutation perturbance element being exactly 0.
        mutator = UniformMutation(rate=1.)
        mutant = mutator.apply(self.sample_individuals)
        self.assertFalse(np.all(mutant == self.sample_individuals))


class TestSelection(unittest.TestCase):

    def setUp(self):
        self.sample_individuals = np.stack([np.zeros(3)]*9 + [np.ones(3) + 1.], axis=0)
        self.sample_grades = np.arange(len(self.sample_individuals), 0, -1)
        self.mate_op = DefaultMate()

    def test_number_of_selected_individuals_corresponds_to_set_selection_rate(self):
        rate = 0.5
        selector = ElitismSelection(selection_rate=rate, mate_op=self.mate_op)
        selector(self.sample_individuals, self.sample_grades)
        self.assertEqual(selector.mask.sum(), rate * len(self.sample_individuals))

    def test_elitism_with_an_engineered_population(self):
        rate = 0.8
        no_offsprings = int(rate * 10)

        selector = ElitismSelection(selection_rate=rate, mate_op=SmoothMate(), exclude_self_mating=True)
        offspring = selector(self.sample_individuals, self.sample_grades, inplace=False)
        self.assertTrue(np.all(offspring[:no_offsprings] == 1.))
