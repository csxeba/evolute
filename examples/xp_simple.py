import numpy as np

from evolute import GeneticPopulation, operators
from evolute.fitness import SimpleFunction
from evolute.utility import describe

TARGET = np.ones(10) * 0.5

pop = GeneticPopulation(loci=10,
                        fitness_wrapper=SimpleFunction(lambda ind: np.linalg.norm(ind - TARGET)),
                        mate_op=operators.SmoothMate())

pop.update()
describe(pop, show=3)
pop.run(100, verbosity=0)
describe(pop, show=3)
