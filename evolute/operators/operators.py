import numpy as np
from . import DefaultSelection, DefaultMutation, DefaultMate


class Operators:

    def __init__(self, selection_op=None, mutate_op=None, mate_op=None):
        self.selection = DefaultSelection() if selection_op is None else selection_op
        self.mutation = DefaultMutation() if mutate_op is None else mutate_op
        self._clarify_mate_operator(mate_op)

    def _clarify_mate_operator(self, mate_op):
        mate_set_in_selection = self.selection.mate_op is not None
        mate_set_here = mate_op is not None
        if mate_set_in_selection and mate_set_here:
            print(" [w] Evolute: differring mate ops, using the one in Selection!")
        elif mate_set_in_selection and not mate_set_here:
            pass
        elif not mate_set_in_selection and mate_set_here:
            self.selection.set_mate_operator(mate_op)
        elif not mate_set_in_selection and not mate_set_here:
            self.selection.set_mate_operator(DefaultMate())
        else:
            assert False, "O.o"  # w00t

    def invalid_individual_indices(self):
        return np.where(self.selection.mask | self.mutation.mask)[0]
