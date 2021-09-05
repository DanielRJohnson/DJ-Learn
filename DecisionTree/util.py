from typing import Union
from collections import Counter
import numpy as np

def class_counts(rows: "list[list]") -> "dict[str, int]":
    return dict(Counter(np.array(rows)[:,-1].flatten()))

def partition(rows: "list[list]", decision) -> Union[list, list]:
    fullfills_decision = np.array([decision.is_match(row) for row in rows])

    true_rows = []
    false_rows = []
    for index, row_fullfills_decision in enumerate(fullfills_decision):
        if row_fullfills_decision:
            true_rows.append(rows[index])
        else:
            false_rows.append(rows[index])

    return true_rows, false_rows

def gini_impurity(rows: "list[list]") -> float:
    counts = class_counts(rows)
    gini = 1
    denom = float(len(rows))
    for label in counts:
        gini -= np.square(counts[label] / denom)
    return gini

def information_gain(left: "list[list]", right: "list[list]", current_uncertainty: float) -> float:
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini_impurity(left) - (1 - p) * gini_impurity(right)
