import numpy as np


class Initializer:

    def initialize(self, *shape):
        raise NotImplementedError


class NormalRandom(Initializer):

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def initialize(self, *shape):
        return np.random.randn(*shape) * self.std + self.mean


class UniformRandom(Initializer):

    def __init__(self, low=-1, high=1.):
        self.low = low
        self.high = high

    def initialize(self, *shape):
        return np.random.uniform(self.low, self.high, shape)


class OrthogonalNormal(Initializer):

    def initialize(self, *shape):
        individials = NormalRandom().initialize(shape)
        d, V = np.linalg.eig(np.cov(individials.T))
        D = np.diag(1. / np.sqrt(d + 1e-7))
        W = V @ D @ V.T
        return individials @ W
