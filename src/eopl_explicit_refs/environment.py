from __future__ import annotations
import abc


class Env(abc.ABC):
    def extend(self, *var: str):
        env = self
        for v in var:
            env = Entry(v, env)
        return env

    @abc.abstractmethod
    def lookup(self, var: str) -> int:
        pass


class EmptyEnv(Env):
    def __init__(self):
        pass

    def lookup(self, var: str) -> int:
        raise LookupError(var)


class Entry(Env):
    def __init__(self, var: str, nxt: Env):
        self.var = var
        self.nxt = nxt

    def lookup(self, var: str) -> int:
        if self.var == var:
            return 0
        return self.nxt.lookup(var) + 1
