import numpy as np

from .grade import SumGrade


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
        self.functions = functions_by_name
        self.kw = {} if constants_by_function_name is None else constants_by_function_name
        self.order = sorted(self.functions) if order_by_name is None else order_by_name
        self.grader = SumGrade() if grader is None else grader

    def __call__(self, phenotype, **variables_by_function):
        fitness = np.array([self.functions[funcname](phenotype, **self.kw[funcname], **variables_by_function[funcname])
                            for funcname in self.order])
        return self.grader(fitness)


class MultiReturnFitness(FitnessBase):

    def __init__(self, fitness_function, number_of_return_values, constants: dict=None, grader=None):
        super().__init__(no_fitnesses=number_of_return_values)
        self.function = fitness_function
        self.constants = {} if constants is None else constants
        self.grader = SumGrade() if grader is None else grader

    def __call__(self, phenotype, **variables):
        fitness = np.array(self.function(phenotype, **self.constants, **variables))
        return self.grader(fitness)
