from __future__ import annotations
import contextlib
from typing import Generic, TypeVar, Optional

T = TypeVar("T")


class Bindings(Generic[T]):
    def __init__(self, key=None, val=None, next=None):
        self.key = key
        self.val = val
        self.next = next

    def depth(self) -> int:
        if self.key is None:
            return 0
        return 1 + self.next.depth()

    def get(self, k: str) -> T:
        match self.key:
            case None:
                raise LookupError(k)
            case key if key == k:
                return self.val
            case _:
                return self.next.get(k)

    def extend(self, k: str, v: T) -> Bindings[T]:
        return Bindings(k, v, self)
