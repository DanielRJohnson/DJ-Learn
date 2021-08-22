import numpy as np

class Cost:
    pass

class ReducedMeanSquaredError(Cost):
    def __init__(self):
        pass
    
    @staticmethod
    def cost(target: np.ndarray, predictions: np.ndarray) -> float:
        return (1/2 * target.shape[0]) * np.sum(np.square(target - predictions))

    @staticmethod
    def derived_cost(target: np.ndarray, predictions: np.ndarray) -> float:
        return predictions - target

class CategoricalCrossentropy(Cost):
    def __init__(self):
        pass

    @staticmethod
    def cost(target: np.ndarray, predictions: np.ndarray) -> float:
        return -np.sum(target * np.log(predictions + 1e-10)) / predictions.shape[0]

    @staticmethod
    def derived_cost(target: np.ndarray, predictions: np.ndarray) -> float:
        return -( (target/predictions) - ( (1 - target) / (1 - predictions) ) )