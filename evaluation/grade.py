import numpy as np


class GradeBase:

    def __call__(self, fitness):
        raise NotImplementedError


class NoopGrade(GradeBase):

    def __call__(self, fitness):
        return fitness


class SumGrade(GradeBase):

    def __call__(self, fitness):
        return fitness.sum()


class WeightedSumGrade(GradeBase):

    def __init__(self, weights=None):
        self.weights = np.ones(1) if weights is None else weights

    def __call__(self, fitness):
        return np.dot(fitness, self.weights)
