from __future__ import annotations
import abc
from typing import TypeVar, Generic

T = TypeVar("T")


class Env(abc.ABC, Generic[T]):
    def extend(self, var: str, val: T):
        return Entry(var, val, self)

    @abc.abstractmethod
    def lookup(self, var: str) -> T:
        pass


class EmptyEnv(Env[T]):
    def __init__(self):
        pass

    def lookup(self, var: str) -> T:
        raise LookupError(var)


class Entry(Env[T]):
    def __init__(self, var: str, val: T, nxt: Env):
        self.var = var
        self.val = val
        self.nxt = nxt

    def lookup(self, var: str) -> T:
        if self.var == var:
            return self.val
        return self.nxt.lookup(var)