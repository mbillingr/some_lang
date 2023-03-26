import dataclasses
from typing import Any


class PythonStore:
    @dataclasses.dataclass
    class Ref:
        val: Any

    def __init__(self):
        self.stack = []

    def clear(self):
        self.stack = []

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        return self.stack.pop()

    def get(self, idx: int):
        return self.stack[idx]

    def is_reference(self, x: Any) -> bool:
        return isinstance(x, PythonStore.Ref)

    def newref(self, val: Any) -> Ref:
        return PythonStore.Ref(val)

    def deref(self, ref: Ref) -> Any:
        return ref.val

    def setref(self, ref: Ref, val: Any):
        ref.val = val
