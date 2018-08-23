import numpy as np

from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical

from evolute import GeneticPopulation
from evolute.evaluation import SimpleFitness
from evolute.initialization import NormalRandom
from evolute.operators import RandomPickMate, Operators, UniformMutation, ScatterMateWrapper
from evolute.utility.keras_utility import WeightFolding


def pull_mnist():
    learning, testing = mnist.load_data()
    Xs, Ys = (learning[0], testing[0]), (learning[1], testing[1])
    Xs = map(lambda x: (x.reshape(-1, 784) - 127.5) / 255., Xs)
    Ys = map(lambda y: to_categorical(y, num_classes=10), Ys)
    return tuple(Xs), tuple(Ys)


def fitness_callback(phenotype, model: Sequential, w_folder, X, Y):
    model.set_weights(w_folder(phenotype))
    cost, acc = model.evaluate(X, Y, verbose=0)
    return cost


(lX, tX), (lY, tY) = pull_mnist()


ann = Sequential([
    Dense(64, activation="tanh", input_dim=lX.shape[1]),
    Dense(lY.shape[1], activation="softmax")
])
ann.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["acc"])

w_shapes = [w.shape for w in ann.get_weights()]
w_flat = np.concatenate([w.flat for w in ann.get_weights()])
w_folder = WeightFolding(ann)

fitness = SimpleFitness(fitness_function=fitness_callback,
                        constants={"model": ann, "w_folder": w_folder})

population = GeneticPopulation(
    limit=100,
    loci=w_flat.size,
    fitness_wrapper=fitness,
    initializer=NormalRandom(mean=w_flat),
    operators=Operators(mate_op=ScatterMateWrapper(RandomPickMate(), stdev=2.),
                        mutate_op=UniformMutation(low=-3., high=3.))
)

BATCH_SIZE = 128
batch_stream = ((lX[start:start+BATCH_SIZE], lY[start:start+BATCH_SIZE])
                for start in range(0, len(lX), BATCH_SIZE))

population.operators.selection.set_selection_rate(0.98)
population.operators.mutation.set_rate(0.0)
for i, (x, y) in enumerate(batch_stream, start=1):
    population.epoch(X=x, Y=y, verbosity=0)
    ann.set_weights(w_folder(population.get_best()))
    cost, acc = ann.evaluate(tX, tY, verbose=0)
    print("\rBatch: {} Acc: {:.2%}, Cost: {:.4f}".format(i, acc, cost))
