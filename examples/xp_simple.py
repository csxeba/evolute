import numpy as np

from matplotlib import pyplot as plt

from evolute import GeneticPopulation
from evolute.evaluation import SimpleFitness

TARGET = np.ones(10) * 0.5

pop = GeneticPopulation(loci=10,
                        fitness_wrapper=SimpleFitness(lambda ind: np.linalg.norm(ind - TARGET)))

history = pop.run(100)
history = {k: np.array(v) for k, v in history.history.items()}

x = history["generation"]

plt.plot(x, history["mean_grade"], "r-", label="mean")
plt.plot(x, history["mean_grade"] + history["grade_std"], "b--", label="std")
plt.plot(x, history["mean_grade"] - history["grade_std"], "b--")
plt.plot(x, history["best_grade"], "g-", label="mean")

plt.title("Population convergence")
plt.legend()
plt.grid()
plt.show()
