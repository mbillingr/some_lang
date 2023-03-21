import dataclasses
from typing import Any, TypeAlias, NewType

Store = NewType("Store", list)


@dataclasses.dataclass
class Ref:
    r: int


def initialize_store():
    _the_store.clear()


def get_store() -> Store:
    return _the_store


def empty_store() -> Store:
    return Store([])


def is_reference(v: Any) -> bool:
    return isinstance(v, Ref)


def newref(val: Any) -> Ref:
    next_ref = len(_the_store)
    _the_store.append(val)
    return Ref(next_ref)


def deref(ref: Ref) -> Any:
    return _the_store[ref.r]


def setref(ref: Ref, val: Any):
    _the_store[ref.r] = val


_the_store: Store = Store([])
