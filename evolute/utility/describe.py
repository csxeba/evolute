import numpy as np


def describe(population, show=0):
    showme = np.argsort(population.grades)[:show]
    chain = "-" * 50 + "\n"
    shln = len(str(show))
    for i, index in enumerate(showme, start=1):
        genomechain = ", ".join(
            "{:>6.4f}".format(loc) for loc in
            np.round(population.individuals[index], 4))
        fitnesschain = "[" + ", ".join(
            "{:^8.4f}".format(fns) for fns in
            population.fitnesses[index]) + "]"
        chain += "TOP {:>{w}}: [{:^14}] F = {:<} G = {:.4f}\n".format(
            i, genomechain, fitnesschain, population.grades[index],
            w=shln)
    best_arg = population.grades.argmin()
    chain += "Best Grade : {:7>.4f} ".format(population.grades[best_arg])
    chain += "Fitnesses: ["
    chain += ", ".join("{}".format(f) for f in population.fitnesses[best_arg])
    chain += "]\n"
    chain += "Mean Grade : {:7>.4f}, STD: {:7>.4f}\n" \
        .format(population.grades.mean(), population.grades.std())
    print(chain)
