import numpy as np
from activations import Activation

class Layer:
    pass

class Dense(Layer):
    def __init__(self, size: int, n_inputs: int, activation: Activation = None):
        self.size = size
        self.n_inputs = n_inputs
        self.w = np.random.randn(self.n_inputs, self.size) / np.sqrt(n_inputs)
        self.b = np.random.randn(1, self.size) / np.sqrt(n_inputs)
        self.activation = activation.activation
        self.derived_activation = activation.derived_activation

    def __call__(self, X: np.ndarray, include_z: bool = False) -> np.ndarray:
        z = X.dot(self.w) + self.b
        if include_z:
            return (self.activation(z), z) if self.activation is not None else (z, z)
        return self.activation(z) if self.activation is not None else z

    def __repr__(self) -> str:
        return f"Activation: {self.activation.__name__}, Size: {self.size}, Inputs: {self.n_inputs}.\nWeights:\n {self.w},\n Biases:\n {self.b}\n"

    def __str__(self) -> str:
        return f"Dense {self.activation.__name__} layer with {self.size} neurons and {self.n_inputs} inputs.\n"