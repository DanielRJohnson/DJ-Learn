from DecisionTree import DecisionTree
import numpy as np
from tqdm import tqdm

class RandomForestClassifier:
    def __init__(self, forest_size: int = 100, max_depth: int = 3) -> None:
        self.max_depth = max_depth
        self.forest = [DecisionTree(max_depth = max_depth) for _ in range(forest_size)]

    def fit(self, X: "list[list]", y: list, rows_subset_fraction = 0.15, pbar = False) -> None:
        # train each tree on 15% of examples, or a fraction specified by the user
        subset_size = int(len(X) * rows_subset_fraction)
        size_range = np.arange(0, len(y))
        for tree in (self.forest if not pbar else tqdm(self.forest)):
            batch_rows = np.random.choice(size_range, size=subset_size)
            tree.fit(X[batch_rows], y[batch_rows])

    def predict(self, rows: "list[list]") -> list:
        # transpose to get from (predictions by tree) to (tree_predictions by example)
        all_tree_predictions = np.transpose([tree.predict(rows) for tree in self.forest])

        # list of tuples like so: (unique_values_in_row, counts_from_each_tree_voting)
        unique_values_and_counts = [np.unique(row, return_counts=True) for row in all_tree_predictions]

        # get the most voted element per row
        final_predictions = [row[0][np.argmax(row[1])] for row in unique_values_and_counts]
        return final_predictions