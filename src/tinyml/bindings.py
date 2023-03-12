from __future__ import annotations
import contextlib
from typing import Generic, TypeVar, Optional

T = TypeVar("T")


class Bindings(Generic[T]):
    def __init__(self):
        self.m: dict[str, T] = {}
        self.changes: list[tuple[str, Optional[T]]] = []

    def get(self, k: str) -> T:
        return self.m[k]

    def insert(self, k: str, v: T) -> Bindings[T]:
        old = self.m.get(k)
        self.changes.append((k, old))
        self.m[k] = v
        return self

    @contextlib.contextmanager
    def child_scope(self):
        n = len(self.changes)
        try:
            yield self
        finally:
            self.unwind(n)

    def unwind(self, n):
        while len(self.changes) > n:
            k, old = self.changes.pop()
            if old is None:
                del self.m[k]
            else:
                self.m[k] = old
