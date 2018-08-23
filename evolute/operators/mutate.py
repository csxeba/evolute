import numpy as np


class MutationBase:

    def __init__(self, rate=0.1):
        self.rate = rate
        self.mask = None

    def noise_source(self, size, seed=None):
        raise NotImplementedError

    def set_rate(self, rate):
        if rate < 0. or rate > 1.:
            raise ValueError("Mutation rate has to be >= 0 and <= 1")
        self.rate = rate

    def apply(self, individuals, inplace=False, seeds=None):
        N, L = individuals.shape
        self._determine_mutation_mask(N)
        if self.rate == 0.:
            return
        if seeds is not None:
            noise = np.array([self.noise_source(size=L, seed=seed) for seed in seeds])
        else:
            noise = self.noise_source(size=(self.mask.sum(), individuals.shape[-1]))
        self._apply_noise_to_individuals(individuals, inplace, noise)
        return individuals

    def _determine_mutation_mask(self, N):
        if self.rate == 0.:
            self.mask = np.zeros(N, dtype=bool)
        elif self.rate == 1.:
            self.mask = np.ones(N, dtype=bool)
        else:
            self.mask = np.random.uniform(size=N) < self.rate

    def _apply_noise_to_individuals(self, individuals, inplace, noise):
        if not inplace:
            mutants = individuals.copy()
            mutants[self.mask] += noise
            return mutants
        individuals[self.mask] += noise

    def __call__(self, individuals, inplace=False, seeds=None):
        return self.apply(individuals, inplace, seeds)


class UniformMutation(MutationBase):

    def __init__(self, rate=0.1, low=-1., high=1.):
        super().__init__(rate)
        self.low = low
        self.high = high

    def noise_source(self, size, seed=None):
        if seed is not None:
            np.random.seed = seed
        return np.random.uniform(low=self.low, high=self.high, size=size)

    def set_params(self, low=None, high=None):
        self.low = self.low if low is None else low
        self.high = self.high if high is None else high


class NormalMutation(MutationBase):

    def __init__(self, rate=0.1, stdev=1.):
        super().__init__(rate)
        self.std = stdev

    def noise_source(self, size, seed=None):
        if seed is not None:
            np.random.seed = seed
        return np.random.normal(loc=0., scale=self.std, size=size)

    def set_param(self, stdev):
        self.std = stdev


DefaultMutation = UniformMutation
