import numpy as np

from matplotlib import pyplot as plt

from brainforge import LayerStack
from brainforge.layers import DenseLayer
from brainforge.cost import cost_factory

from evolute.operators import ScatterMateWrapper, SmoothMate, Operators
from evolute import GeneticPopulation

np.random.seed(1234)

rX = np.linspace(-6., 6., 200)[:, None]
rY = np.sin(rX)

arg = np.arange(len(rX))
np.random.shuffle(arg)
targ, varg = arg[:100], arg[100:]
targ.sort()
varg.sort()

tX, tY = rX[targ], rY[targ]
vX, vY = rX[varg], rY[varg]

tX += np.random.randn(*tX.shape) / np.sqrt(tX.size*0.25)

loss_fn = cost_factory("mse")


def fitness(phenotype, layerstack, X, Y):
    layerstack.set_weights(phenotype)
    return loss_fn(layerstack.predict(X), Y)


def forge_layerstack():
    return LayerStack(input_shape=(1,), layers=[
        DenseLayer(30, activation="tanh"),
        DenseLayer(30, activation="tanh"),
        DenseLayer(1, activation="linear")
    ])


def get_population():
    layers = forge_layerstack()
    operators = Operators(mate_op=ScatterMateWrapper(SmoothMate, 3.))
    pop = GeneticPopulation.simple_fitness(limit=100, loci=layers.nparams,
                                           operators=operators,
                                           fitness_callback=fitness,
                                           fitness_constants={"layerstack": layers})
    return layers, pop


def xperiment():
    layers, pop = get_population()
    layers = forge_layerstack()
    tpred = layers.predict(tX)
    vpred = layers.predict(vX)
    plt.ion()
    plt.plot(tX, tY, "b--", alpha=0.5, label="Training data (noisy)")
    plt.plot(rX, rY, "r--", alpha=0.5, label="Validation data (clean)")
    plt.ylim(min(rY)-1, max(rY)+1)
    plt.plot(rX, np.zeros_like(rX), c="grey", linestyle="--")
    tobj, = plt.plot(tX, tpred, "bo", markersize=3, alpha=0.5, label="Training pred")
    vobj, = plt.plot(vX, vpred, "ro", markersize=3, alpha=0.5, label="Validation pred")
    templ = "Batch: {:>5} Cost = {:.4f}"
    t = plt.title(templ.format(0, 0))
    plt.legend()
    batchno = 1
    while 1:
        pop.epoch(X=tX, Y=tY)
        layers.set_weights(pop.get_champion())
        tpred = layers.predict(tX)
        vpred = layers.predict(vX)
        tobj.set_data(tX, tpred)
        vobj.set_data(vX, vpred)
        plt.pause(0.01)
        t.set_text(templ.format(batchno, pop.fitnesses.min()))
        batchno += 1


if __name__ == '__main__':
    xperiment()
