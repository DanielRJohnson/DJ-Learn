import numpy as np

class Activation:
    pass

class Softmax(Activation):
    def __init__(self):
        pass
    
    @staticmethod
    def activation(inp: np.ndarray) -> float:
        inp_norm = inp - inp.max()
        exp = np.exp(inp_norm)
        return exp / exp.sum(axis=1, keepdims=True)

    @staticmethod
    def derived_activation(inp: np.ndarray) -> float:
        inp2d = inp.reshape(-1, 1)
        return np.diagflat(inp2d) - np.dot(inp2d, inp2d.T)

class Sigmoid(Activation):
    def __init__(self):
        pass
    
    @staticmethod
    def activation(inp: np.ndarray) -> float:
        inp_norm = inp - inp.max()
        return 1 / (1 + np.exp(-inp_norm))

    @staticmethod
    def derived_activation(inp: np.ndarray) -> float:
        return Sigmoid.activation(inp) * (1 - Sigmoid.activation(inp))

class Tanh(Activation):
    def __init__(self):
        pass
    
    @staticmethod
    def activation(inp: np.ndarray) -> float:
        inp_norm = inp - inp.max()
        exp = np.exp(inp_norm)
        nexp = np.exp(-inp_norm)
        return (exp - nexp) / (exp + nexp)

    @staticmethod
    def derived_activation(inp: np.ndarray) -> float:
        return 1 - (np.square(Tanh.activation(inp)))

class ReLu(Activation):
    def __init__(self):
        pass
    
    @staticmethod
    def activation(inp: np.ndarray) -> float:
        return np.maximum(inp, 0)

    @staticmethod
    def derived_activation(inp: np.ndarray) -> float:
        return np.greater(inp, 0)