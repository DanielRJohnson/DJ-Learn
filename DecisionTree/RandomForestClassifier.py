from DecisionTree import DecisionTree
import numpy as np
from tqdm import tqdm

#TODO: switch completely to numpy, preformance is pretty bad

class RandomForestClassifier:
    def __init__(self, forest_size: int = 100, max_depth: int = 3) -> None:
        self.max_depth = max_depth
        self.forest = [DecisionTree(max_depth = max_depth) for _ in range(forest_size)]

    def fit(self, X: "list[list]", y: list) -> None:
        for tree in tqdm(self.forest):
            tree.fit(X, y)

    def predict(self, rows: "list[list]") -> list:
        predictions_by_tree = np.transpose([tree.predict(rows) for tree in self.forest]).tolist()
        final_predictions = [max(set(pred), key = pred.count) for pred in predictions_by_tree]
        return final_predictions