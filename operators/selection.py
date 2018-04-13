import numpy as np

from .mate import DefaultMate


class SelectionBase:

    def __init__(self, selection_rate=0.5, mate_op=None, exclude_self_mating=True):
        self.mate_op = None
        self.rate = None
        self._selection_mask = None
        self.exclude_self_mating = exclude_self_mating
        self.set_mate_operator(DefaultMate() if mate_op is None else mate_op)
        self.set_selection_rate(selection_rate)

    @property
    def mask(self):
        return self._selection_mask

    def set_mate_operator(self, mate_op):
        self.mate_op = mate_op

    def _stream_of_parent_indices(self):
        assert self._selection_mask is not None
        survivor_mask = ~self._selection_mask
        arg1 = np.argwhere(survivor_mask)[:, 0]
        assert len(arg1) > 1
        arg2 = np.copy(arg1)
        limit = sum(self._selection_mask)
        n = 0
        while n < limit:
            np.random.shuffle(arg1)
            np.random.shuffle(arg2)
            for ix1, ix2 in zip(arg1, arg2):
                if n >= limit:
                    raise StopIteration
                if self.exclude_self_mating and ix1 == ix2:
                    continue
                yield ix1, ix2
                n += 1

    def set_survival_rate(self, survival_rate):
        if survival_rate <= 0. or survival_rate > 1.:
            raise ValueError("The rate of survival has to be greater than 0 and less or equal to 1")
        self.rate = survival_rate

    def set_selection_rate(self, selection_rate):
        if selection_rate <= 0. or selection_rate > 1.:
            raise ValueError("The rate of selection has to be greater than 0 and less or equal to 1")
        self.rate = 1. - selection_rate

    def apply(self, individuals, grades, inplace=False):
        raise NotImplementedError

    def __call__(self, individuals, grades, inplace=False):
        return self.apply(individuals, grades, inplace)


class Elitism(SelectionBase):

    def apply(self, individuals, grades, inplace=False):
        self._selection_step(individuals, grades)
        if not inplace:
            return self._reproduction_copy(individuals)
        self._reproduction_inplace(individuals)

    def _selection_step(self, individuals, grades):
        limit, loci = individuals.shape
        self._selection_mask = np.ones(limit, dtype=bool)
        if self.rate:
            no_survivors = max(2, int(limit * self.rate))
            survivors = np.argsort(grades)[:no_survivors]
            self._selection_mask[survivors] = False

    def _reproduction_inplace(self, individuals):
        individuals[self._selection_mask] = [
            self.mate_op(individuals[idx1], individuals[idx2])
            for idx1, idx2 in self._stream_of_parent_indices()
        ]

    def _reproduction_copy(self, individuals):
        offspring = individuals.copy()
        new_indivs = [
            self.mate_op(offspring[idx1], offspring[idx2])
            for idx1, idx2 in self._stream_of_parent_indices()
        ]
        offspring[self._selection_mask] = new_indivs
        return offspring


DefaultSelection = Elitism
