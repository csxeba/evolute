import numpy as np


class MateBase:

    def apply(self, ind1, ind2):
        pass

    def __call__(self, ind1, ind2):
        return self.apply(ind1, ind2)


class LambdaMate(MateBase):

    def __init__(self, function_ref, **kw):
        self.kwargs = kw
        self.apply = lambda ind1, ind2: function_ref(ind1, ind2, **self.kwargs)


class RandomPickMate(MateBase):

    def apply(self, ind1, ind2):
        return np.where(np.random.uniform(size=ind1.shape) < 0.5, ind1, ind2)


class SmoothMate(MateBase):

    def apply(self, ind1, ind2):
        return np.mean((ind1, ind2), axis=0)


DefaultMate = RandomPickMate


class ScatterMateWrapper(MateBase):

    def __init__(self, base=DefaultMate, sigma=1.):
        if isinstance(base, type):
            base = base()
        self.base = base
        self.sigma = sigma

    def apply(self, ind1, ind2):
        return self.base(ind1, ind2) + np.random.randn(*ind1.shape) * self.sigma
