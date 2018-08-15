import numpy as np


class GraderBase:

    def __call__(self, fitness):
        raise NotImplementedError


class SumGrader(GraderBase):

    def __call__(self, fitness):
        return np.sum(fitness)


class WeightedSumGrader(GraderBase):

    def __init__(self, weights):
        self.weights = np.ones(1) if weights is None else weights

    def __call__(self, fitness):
        return np.dot(fitness, self.weights)
