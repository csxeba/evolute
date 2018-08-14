import numpy as np
from . import DefaultSelection, DefaultMutate, DefaultMate


class Operators:

    def __init__(self, selection_op=None, mutate_op=None, mate_op=None):
        """
        :param selection_op: selection operator,
        :param mate_op: accepts two genotypes, returns an offspring genotype
        :param mutate_op: accepts individuals and mutation rate, returns mutants
        """
        self.selection = DefaultSelection() if selection_op is None else selection_op
        self.mutation = DefaultMutate() if mutate_op is None else mutate_op
        self.mate = DefaultMate() if mate_op is None else mate_op
        self.selection.set_mate_operator(self.mate)

    def invalid_individual_indices(self):
        return np.where(self.selection.mask | self.mutation.mask)[0]
