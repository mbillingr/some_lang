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
        return Entry(var, self)

    def lookup(self, var: str) -> int:
        raise LookupError(var)


class Entry(Env):
    def __init__(self, var: str, nxt: Env):
        self.var = var
        self.nxt = nxt

    def extend(self, var: str):
        return Entry(var, self)

    def lookup(self, var: str) -> int:
        if self.var == var:
            return 0
        return self.nxt.lookup(var) + 1
