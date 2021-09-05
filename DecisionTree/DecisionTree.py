from typing import Union

from DecisionNodes import DecisionNode, DecisionLeaf, Decision
from util import partition, gini_impurity, information_gain

class DecisionTree:
    def __init__(self, max_depth: int = 3) -> None:
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: "list[list]", y: list) -> None:
        rows = [list(row) + [y_val] for row, y_val in zip(list(X),list(y))]
        self.root = self._build(rows)

    def predict(self, rows: "list[list]") -> list:
        prediction_counts = [self._predict(row, self.root) for row in rows]
        return [max(count, key=count.get) for count in prediction_counts]

    def _predict(self, row: list, node: Union[DecisionNode, DecisionLeaf]) -> "dict[str, int]":
        if isinstance(node, DecisionLeaf):
            return node.predictions
        
        if node.decision.is_match(row):
            return self._predict(row, node.true_branch)
        else:
            return self._predict(row, node.false_branch)

    def _build(self, rows: "list[list]", depth: int = 0):
        gain, decision = self._find_best_split(rows)

        if gain == 0 or depth >= self.max_depth:
            return DecisionLeaf(rows)

        true_rows, false_rows = partition(rows, decision)

        true_branch = self._build(true_rows, depth + 1)
        false_branch = self._build(false_rows, depth + 1)

        return DecisionNode(decision, true_branch, false_branch)

    def _find_best_split(self, rows: "list[list]") -> Union[float, Decision]:
        best_gain = 0
        best_decision = None
        current_uncertainty = gini_impurity(rows)
        n_features = len(rows[0]) - 1

        for col in range(n_features):
            values = set([row[col] for row in rows])
            for value in values:
                decision = Decision(col, value)
                true_rows, false_rows = partition(rows, decision)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                gain = information_gain(true_rows, false_rows, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_decision = gain, decision
                
        return best_gain, best_decision