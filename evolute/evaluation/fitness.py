import numpy as np

from .grade import SumGrader


class FitnessBase:

    def __init__(self, no_fitnesses):
        self.no_fitnesses = no_fitnesses

    def __call__(self, phenotype):
        raise NotImplementedError


class SimpleFitness(FitnessBase):

    def __init__(self, fitness_function, constants: dict=None, **kw):
        super().__init__(no_fitnesses=1)
        self.function = fitness_function
        self.constants = {} if constants is None else constants
        self.constants.update(kw)

    def __call__(self, phenotype, **variables):
        return self.function(phenotype, **self.constants, **variables)


class MultipleFitnesses(FitnessBase):

    def __init__(self, functions_by_name, constants_by_function_name=None, order_by_name=None, grader=None):
        super().__init__(no_fitnesses=len(functions_by_name))
        if len(functions_by_name) < 2:
            raise ValueError("MultipleFitnesses needs more than one fitness!")
        self.functions = functions_by_name
        self.order = order_by_name or list(self.functions)
        self.constants = constants_by_function_name or {k: {} for k in self.order}
        self.grader = grader or SumGrader()
        if len(self.order) != len(self.functions) or any(o not in self.functions for o in self.order):
            raise ValueError("The specified order is wrong: {}".format(self.order))
        if len(self.constants) != len(self.functions) or any(k not in self.functions for k in self.constants):
            raise ValueError("The specified constants are wrong: {}".format(self.constants))

    def __call__(self, phenotype, **variables_by_function):
        fitness = np.array(
            [self.functions[funcname](phenotype, **self.constants[funcname], **variables_by_function[funcname])
             for funcname in self.order])
        return self.grader(fitness)


class MultiReturnFitness(FitnessBase):

    def __init__(self, fitness_function, number_of_return_values, constants: dict=None, grader=None):
        super().__init__(no_fitnesses=number_of_return_values)
        self.function = fitness_function
        self.constants = {} if constants is None else constants
        self.grader = SumGrader() if grader is None else grader

    def __call__(self, phenotype, **variables):
        fitness = np.array(self.function(phenotype, **self.constants, **variables))
        return self.grader(fitness)
