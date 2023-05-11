import dataclasses
from typing import Any, TypeVar, Generic

CLS = TypeVar("CLS")


class PythonStore(Generic[CLS]):
    @dataclasses.dataclass
    class Ref:
        val: Any

    def __init__(self):
        self.env = ()
        self.methods = []
        self.vtables = []

    def clear(self):
        self.env = ()
        self.methods = []
        self.vtables = []

    def push(self, *vals):
        self.env = (list(vals), self.env)

    def pop(self):
        self.env = self.env[1]

    def get(self, idx: tuple[int, int]):
        ofs, depth = idx
        frame = self.env
        for _ in range(depth):
            frame = frame[1]
        return frame[0][ofs]

    def set(self, idx: int, val):
        ofs, depth = idx
        frame = self.env
        for _ in range(depth):
            frame = self.env[1]
        frame[0][ofs] = val

    def is_reference(self, x: Any) -> bool:
        return isinstance(x, PythonStore.Ref)

    def newref(self, val: Any) -> Ref:
        return PythonStore.Ref(val)

    def deref(self, ref: Ref) -> Any:
        return ref.val

    def setref(self, ref: Ref, val: Any):
        ref.val = val

    def add_method(self, method: CLS):
        return self.methods.append(method)

    def get_method(self, idx: int) -> CLS:
        return self.methods[idx]

    def set_vtables(self, vtables):
        self.vtables = vtables

    def get_vtable(self, vtidx: int):
        return self.vtables[vtidx]
