import numpy as np
from csxdata.utilities.loader import pull_mnist_data
from evolute import GeneticPopulation
from evolute.evaluation import SimpleFitness
from evolute.initialization import NormalRandom
from evolute.operators import SmoothMate, ScatterMateWrapper, Operators
from evolute.utility.keras_utility import WeightFolding
from keras.layers import Dense
from keras.models import Sequential


def fitness_callback(phenotype, model: Sequential, w_folder, X, Y):
    model.set_weights(w_folder(phenotype))
    cost, acc = model.train_on_batch(X, Y)
    return cost


lX, lY, tX, tY = pull_mnist_data(split=0.2, fold=False)


ann = Sequential([
    Dense(60, activation="tanh", input_dim=lX.shape[1]),
    Dense(lY.shape[1], activation="softmax")
])
ann.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["acc"])

w_shapes = [w.shape for w in ann.get_weights()]
w_flat = np.concatenate([w.flat for w in ann.get_weights()])
w_folder = WeightFolding(ann)

fitness = SimpleFitness(fitness_function=fitness_callback,
                        constants={"model": ann, "w_folder": w_folder})

population = GeneticPopulation(
    loci=w_flat.size,
    fitness_wrapper=fitness,
    initializer=NormalRandom(mean=w_flat),
    operators=Operators(mate_op=ScatterMateWrapper(SmoothMate(), sigma=3.))
)

BATCH_SIZE = 128
batch_stream = ((lX[start:start+BATCH_SIZE], lY[start:start+BATCH_SIZE])
                for start in range(0, len(lX), BATCH_SIZE))

population.operators.selection.set_selection_rate(0.8)
population.operators.mutation.set_rate(0.1)
for i, (x, y) in enumerate(batch_stream, start=1):
    population.epoch(X=x, Y=y)
    ann.set_weights(population.get_best())
    cost, acc = ann.evaluate(tX, tY, verbose=0)
    print("\rBatch: {} Acc: {}".format(i, acc))
