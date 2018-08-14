import numpy as np


def get_keras_weights(model, folded=False):
    w_tensors = model.trainable_weights
    if folded:
        return w_tensors
    return np.concatenate([w.flat for w in w_tensors])


def get_keras_number_of_trainables(model):
    return sum(w.size for w in model.trainable_weights)


class WeightFolding:

    def __init__(self, model):
        self.shapes = [w.shape for w in model.get_weights()]
        self.sizes = [np.prod(shape) for shape in self.shapes]

    def __call__(self, individual):
        phenotype = []
        start = 0
        for shape, size in zip(self.shapes, self.sizes):
            end = start + size
            phenotype.append(individual[start:end].reshape(shape))
            start = end
        return phenotype
