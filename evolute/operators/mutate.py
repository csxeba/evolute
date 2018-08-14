import operator

import numpy as np


class MutationBase:

    def __init__(self, rate=0.1):
        self.rate = rate
        self.mask = None

    def set_rate(self, rate):
        if rate < 0. or rate > 1.:
            raise ValueError("Mutation rate has to be >= 0 and <= 1")
        self.rate = rate

    def apply(self, individuals, inplace=False):
        raise NotImplementedError

    def __call__(self, individuals, inplace=False):
        return self.apply(individuals)


class UniformLocuswiseMutation(MutationBase):

    def __init__(self, rate=0.1, low=-1., high=1.):
        super().__init__(rate)
        self.low = low
        self.high = high

    def set_params(self, low=None, high=None):
        self.low = self.low if low is None else low
        self.high = self.high if high is None else high

    def apply(self, individuals, inplace=False):
        indshape = individuals.shape
        if self.rate == 0.:
            mask = np.zeros(indshape, dtype=bool)
        elif self.rate == 1.:
            mask = np.ones(indshape, dtype=bool)
        else:
            mask = np.random.uniform(size=indshape) < (self.rate / indshape[-1])
        self.mask = np.any(mask, axis=1)
        noise = np.random.uniform(self.low, self.high, size=mask.sum())
        if not inplace:
            mutants = individuals.copy()
            mutants[mask] += noise
            return mutants
        individuals[mask] += noise


class NormalIndividualwiseMutation(MutationBase):

    def __init__(self, rate=0.1, stdev=1.):
        super().__init__(rate)
        self.std = stdev

    def set_param(self, stdev):
        self.std = stdev

    def apply(self, individuals, inplace=False):
        mask = np.random.uniform(size=len(individuals)) < self.rate
        noise = np.random.normal(loc=0., scale=self.std, size=(mask.sum(), individuals.shape[-1]))
        if not inplace:
            mutants = individuals.copy()
            mutants[mask] += noise
            return mutants
        individuals[mask] += noise
        self.mask = mask


class BitwiseMutation(MutationBase):

    def __init__(self, bitwise_op="xor", rate=0.1):
        super().__init__(rate)
        self.bitwise_operator = {
            "xor": operator.xor, "and": operator.and_, "or": operator.or_,
            "nand": lambda a, b: operator.not_(operator.and_(a, b)),
            "xnor": lambda a, b: operator.not_(operator.xor(a, b))
        }[bitwise_op]

    def apply(self, individuals, inplace=False):
        limit, loci = individuals.shape
        mask = np.random.uniform(size=limit) < self.rate
        bytestream = np.frombuffer(individuals[mask].tobytes(), dtype="int8")
        noisestream = np.random.randint(-128, 127, size=mask.sum(), dtype="int8")
        mutants = np.bitwise_xor(bytestream, noisestream)  # XOR, so number of 1s and 0s don't change
        mutants_float = np.frombuffer(mutants.tobytes(), dtype=individuals.dtype)
        mutants_float = mutants_float.reshape(len(mutants_float) // loci, loci)
        if not inplace:
            new_individuals = individuals.copy()
            new_individuals[mask] = mutants_float
            return new_individuals
        individuals[mask] = mutants_float


DefaultMutate = UniformLocuswiseMutation
