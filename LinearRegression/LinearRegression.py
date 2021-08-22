import numpy as np
from tqdm import tqdm

class LinearRegression:
    def __init__(self, n_features: int, theta: np.ndarray = None, add_intercept: bool = False):
        #create a new feature that is always one to add bias
        self.theta = np.random.rand(n_features + 1 if add_intercept else n_features, 1) / np.sqrt(n_features) if theta is None else theta
        self.add_intercept = add_intercept
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X ))
        #h(X) = X @ theta 
        return np.dot(X, self.theta)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        #h(X) = X @ theta 
        return np.dot(X, self.theta)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate=0.0001) -> list:
        if self.add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X ))

        cost_history = []
        progress_bar = tqdm(range(epochs))
        for i in progress_bar:
            # theta = theta - alpha * dJ/dw
            # dJ/dw = (1/m) * X.T(h(x) - y)
            error = self._predict(X) - y
            self.theta = self.theta - (learning_rate / y.shape[0]) * np.dot(X.T, error)

            # J = (1/2*m) sum(i = 1 to n) of { (yi - yi(hat))^2 }
            cost_history.append(float(1 / (2 * y.shape[0])  * np.dot(error.T, error)))
            progress_bar.set_description("Cost: " + str(round(cost_history[i], 2)))
        return cost_history