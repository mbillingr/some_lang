from typing import Iterable, TypeAlias

VtableIndex: TypeAlias = tuple[int, int]


class VtableManager:
    def __init__(self):
        self.n_virtuals = 0
        self.n_methods = 0

    def assign_virtuals(self, method_names: Iterable[str]) -> dict[str, VtableIndex]:
        table = 0  # only use a single large table for now
        mapping = {}
        for m in method_names:
            mapping[m] = table, self.n_virtuals
            self.n_virtuals += 1
        return mapping
