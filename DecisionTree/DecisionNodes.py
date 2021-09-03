from typing import Union
from util import class_counts

class Decision:
    def __init__(self, column_index: int, value: Union[int, float, str]) -> None:
        self.column_index = column_index
        self.value = value

    def is_match(self, row: list) -> bool:
        value = row[self.column_index]
        if isinstance(value, int) or isinstance(value, float):
            return value >= self.value
        else:
            return value == self.value

class DecisionNode:
    def __init__(self, decision: Decision, true_branch, false_branch):
        self.decision = decision
        self.true_branch = true_branch
        self.false_branch = false_branch

class DecisionLeaf:
    def __init__(self, rows: "list[list]") -> None:
        self.predictions = class_counts(rows)