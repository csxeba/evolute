import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from csxdata.utilities.loader import pull_mnist_data

from evolute import DifferentialEvolution
from evolute.fitness import SimpleFunction
from evolute.phenotyper import CompoundPhenotyper, ShapePhenotyper, ScalePhenotyper
from evolute.initializer import NormalRandom


def fitness_callback(phenotype, model: Sequential, X, Y):
    model.set_weights(phenotype)
    return model.train_on_batch(X, Y)


lX, lY, tX, tY = pull_mnist_data(split=0.2, fold=False)


ann = Sequential([
    Dense(60, activation="tanh", input_dim=lX.shape[1]),
    Dense(lY.shape[1], activation="softmax")
])
ann.compile(optimizer="sgd", loss="categorical_crossentropy")

w_shapes = [w.shape for w in ann.get_weights()]
w_flat = np.concatenate([w.flat for w in ann.get_weights()])

population = DifferentialEvolution(
    loci=w_flat.size,
    fitness_wrapper=SimpleFunction(fitness_callback, constants={"model": ann}),
    phenotyper=CompoundPhenotyper(ShapePhenotyper(), ScalePhenotyper()),
    initializer=NormalRandom(mean=w_flat)
)

for i, (x, y) in enumerate((lX[start:start+512], lY[start:start+512]) for start in range(0, len(lX), 512)):
    population.epoch(X=x, Y=y)
    ann.set_weights(population.get_best(as_phenotype=True))
    cost, acc = ann.evaluate(tX, tY, verbose=0)
    print("\rBatch: {} Acc: {}".format(i+1, acc))
