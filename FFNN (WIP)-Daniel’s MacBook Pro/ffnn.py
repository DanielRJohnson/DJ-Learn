import numpy as np
from tqdm import tqdm
from functools import reduce
from typing import Union

from layers import Layer
from costs import Cost

class FFNN:
    def __init__(self, layers: "list[Layer]", cost: Cost):
        #maybe should assert valid shapes
        self.layers = [layer for layer in layers]
        self.cost = cost.cost
        self.derived_cost = cost.derived_cost

    def forward(self, rows: np.ndarray) -> np.ndarray:
        return np.vstack([reduce(lambda row, layer: layer(row), self.layers, row) for row in rows])

    #todo: improve this
    def elaborate_forward(self, rows: np.ndarray):
        a_hist, z_hist = [], []
        a_hist.append(rows)
        #z_hist.append(rows)

        for layer_index, layer in enumerate(self.layers):
            z = np.dot(a_hist[layer_index], layer.w) + layer.b
            z_hist.append(z)
            a_hist.append(layer.activation(z))
        return a_hist, z_hist

    def backward(self, rows: np.ndarray, targets: np.ndarray) -> Union[list, list]:
        a_hist, z_hist = self.elaborate_forward(rows)
        deltas = [None] * len(self.layers)

        deltas[-1] = self.derived_cost(targets, a_hist[-1]) * self.layers[-1].derived_activation(z_hist[-1])
        for layer_index, layer in reversed(list(enumerate(self.layers[:-1]))):
            deltas[layer_index] = np.dot( deltas[layer_index + 1], self.layers[layer_index + 1].w.T ) * \
                            layer.derived_activation(z_hist[layer_index])  

        weight_gradient, bias_gradient = [], []
        n_features = rows.shape[0]
        for layer_index in range(len(self.layers)):
            #(1 / n_features) * 
            #maybe a_hist[layer_index - 1]
            weight_gradient.append((1 / n_features) * np.dot( a_hist[layer_index].T, deltas[layer_index]) )
            bias_gradient.append((1 / n_features) * np.sum(deltas[layer_index], axis=0))
        return weight_gradient, bias_gradient

    def train(self, rows: np.ndarray, targets: np.ndarray, epochs: int, learning_rate: float = 0.001, convergence_treshhold: float = None) -> "list[int]":
        costs = []
        i = 0
        progress_bar = tqdm(total=epochs)
        while i < epochs and (convergence_treshhold is None or (i <= 1) or (costs[-2] - costs[-1] >= convergence_treshhold)):
            progress_bar.update(1)
            weight_gradient, bias_gradient = self.backward(rows, targets)
            for layer_index, layer in enumerate(self.layers):
                layer.w = layer.w - learning_rate * weight_gradient[layer_index]
                layer.b = layer.b - learning_rate * bias_gradient[layer_index]
            costs.append(self.cost(targets, self.forward(rows)))
            i += 1
        progress_bar.close()
        return costs

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        return self.forward(rows)

    def __repr__(self) -> str:
        return "".join(list(map(lambda layer: f"Layer {self.layers.index(layer)}:\n(\n{repr(layer)})\n\n", self.layers)))

    def __str__(self) -> str:
        return "".join(list(map(lambda layer: f"Layer {self.layers.index(layer)}: {str(layer)}", self.layers)))