from __future__ import annotations
import abc
from typing import TypeVar, Generic, Iterator

T = TypeVar("T")


class Env(abc.ABC, Generic[T]):
    def extend(self, var: str, val: T):
        return Entry(var, val, self)

    @abc.abstractmethod
    def lookup(self, var: str) -> T:
        pass

    @abc.abstractmethod
    def set(self, var: str, val: T):
        pass

    @abc.abstractmethod
    def items(self) -> Iterator[tuple[str, T]]:
        pass


class EmptyEnv(Env[T]):
    def __init__(self):
        pass

    def lookup(self, var: str) -> T:
        raise LookupError(var)

    def set(self, var: str, val: T):
        raise LookupError(var)

    def items(self):
        return
        yield ()

    def __repr__(self):
        return "()"


class Entry(Env[T]):
    def __init__(self, var: str, val: T, nxt: Env):
        self.var = var
        self.val = val
        self.nxt = nxt

    def lookup(self, var: str) -> T:
        if self.var == var:
            return self.val
        return self.nxt.lookup(var)

    def set(self, var: str, val: T):
        if self.var == var:
            self.val = val
        else:
            self.nxt.set(var, val)

    def items(self):
        yield self.var, self.val
        yield from self.nxt.items()

    def __repr__(self):
        return f"(({self.var} : {self.var}) .\n {self.nxt})"
