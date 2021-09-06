import numpy as np

class kNNClassifier:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> None:
        # X, y -> rows of (x1, x2, ... , xn, y)
        self.stored_data = np.hstack((X, y.reshape(-1, 1)))
        self.n_neighbors = n_neighbors

    def predict(self, X: np.ndarray) -> list:
        predictions = []
        for row in X:
            # for each row in stored data, take the (n_neighbors) shortest distance stored_rows from row 
            neighbors = np.array(sorted(self.stored_data, key=lambda stored_row: np.linalg.norm(stored_row[:-1]-row))[:self.n_neighbors])
            # the k nearest neighbors do a simple vote on the output class
            unique_classes, counts = np.unique(neighbors[:,-1], return_counts=True)
            predictions.append(unique_classes[counts.argmax()])
        return predictions