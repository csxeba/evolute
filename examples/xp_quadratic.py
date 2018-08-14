import numpy as np
from matplotlib import pyplot as plt

from evolute import GeneticPopulation
from evolute.evaluation import SimpleFitness


def fitness(ind, target):
    return np.linalg.norm(target - ind)


def main():
    TARGET = np.array([3., 3.])

    pop = GeneticPopulation(
        loci=2,
        fitness_wrapper=SimpleFitness(fitness, constants={"target": TARGET}),
        limit=100)

    plt.ion()
    obj = plt.plot(*pop.individuals.T, "bo", markersize=2)[0]
    plt.xlim([-2, 11])
    plt.ylim([-2, 11])

    X, Y = np.linspace(-2, 11, 50), np.linspace(-2, 11, 50)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([fitness(np.array([x, y]), target=TARGET)
                  for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
    CS = plt.contour(X, Y, Z, cmap="hot")
    plt.clabel(CS, inline=1, fontsize=10)
    title_template = "Best: [{:.4f}, {:.4f}], G: {:.4f}"
    title_obj = plt.title(title_template.format(0., 0., 0.))
    plt.show()
    means, stds, bests = [], [], []
    for i in range(30):
        pop.epoch(force_update=True, verbosity=0)
        means.append(pop.fitnesses.mean())
        stds.append(pop.fitnesses.std())
        bests.append(pop.fitnesses.min())
        obj.set_data(*pop.individuals.T)
        title_obj.set_text(title_template.format(*pop.get_best(), pop.fitnesses.min()))
        plt.pause(0.1)

    means, stds, bests = tuple(map(np.array, (means, stds, bests)))
    plt.close()
    plt.ioff()
    Xs = np.arange(1, len(means) + 1)
    plt.plot(Xs, means, "b-")
    plt.plot(Xs, means+stds, "g--")
    plt.plot(Xs, means-stds, "g--")
    plt.plot(Xs, bests, "r-")
    plt.xlim([Xs.min()-1, Xs.max()+1])
    plt.ylim([bests.min()-1, (means+stds).max()+1])
    plt.show()


if __name__ == '__main__':
    main()
