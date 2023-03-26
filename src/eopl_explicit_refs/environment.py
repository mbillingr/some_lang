from __future__ import annotations
import abc


class Env(abc.ABC):
    @abc.abstractmethod
    def extend(self, var: str) -> Env:
        pass

    @abc.abstractmethod
    def lookup(self, var: str) -> int:
        pass


class EmptyEnv(Env):
    def __init__(self):
        pass

    def extend(self, var: str):
        return Entry(var, 0, self)

    def lookup(self, var: str) -> int:
        raise LookupError(var)


class Entry(Env):
    def __init__(self, var: str, idx: int, nxt: Env):
        self.var = var
        self.idx = idx
        self.nxt = nxt

    def extend(self, var: str):
        return Entry(var, self.idx + 1, self)

    def lookup(self, var: str) -> int:
        if self.var == var:
            return self.idx
        return self.nxt.lookup(var)
