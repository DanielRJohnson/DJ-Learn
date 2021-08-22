import numpy as np
from numpy.core.defchararray import add
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, n_features: int, theta: np.ndarray = None, add_intercept: bool = False):
        #create a new feature that is always one to add bias
        self.theta = np.random.rand(n_features + 1 if add_intercept else n_features, 1) / np.sqrt(n_features) if theta is None else theta
        self.add_intercept = add_intercept
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X ))
        #h(X) = sigmoid(X @ Theta), round for public output
        scores = np.dot(X, self.theta)
        return np.round(1 / (1 + np.exp(-scores)))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        #h(X) = sigmoid(X @ Theta)
        scores = np.dot(X, self.theta)
        return 1 / (1 + np.exp(-scores))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate=0.0001) -> list:
        if self.add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X ))
            
        likelihood_history = []
        progress_bar = tqdm(range(epochs))
        for i in progress_bar:
            predictions = self._predict(X)

            # theta = theta - alpha * dJ/dw
            # dJ/dw = X.T(y - h(x))
            error = y - predictions
            self.theta = self.theta + learning_rate * np.dot(X.T, error)

            likelihood_history.append(self.log_likelyhood(X, y))
            progress_bar.set_description("Ll: " + str(round(likelihood_history[i], 2)))
        return likelihood_history
    
    def log_likelyhood(self, X: np.ndarray, y: np.ndarray) -> float:
        # Ll = Sum(i = 1 to n) of { yi * theta.T * xi - log(1 + e^(theta.T * xi)) }
        scores = np.dot(X, self.theta)
        return np.sum(y * scores - np.log(1 + np.exp(scores)))