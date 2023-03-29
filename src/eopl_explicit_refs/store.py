import dataclasses
from typing import Any


class PythonStore:
    @dataclasses.dataclass
    class Ref:
        val: Any

    def __init__(self):
        self.stack = ()

    def clear(self):
        self.stack = ()

    def push(self, val):
        self.stack = [val, self.stack]

    def pop(self):
        item = self.stack[0]
        self.stack = self.stack[1]
        return item

    def get(self, idx: int):
        frame = self.stack
        for _ in range(idx):
            frame = self.stack[1]
        return frame[0]

    def set(self, idx: int, val):
        frame = self.stack
        for _ in range(idx):
            frame = self.stack[1]
        frame[0] = val

    def is_reference(self, x: Any) -> bool:
        return isinstance(x, PythonStore.Ref)

    def newref(self, val: Any) -> Ref:
        return PythonStore.Ref(val)

    def deref(self, ref: Ref) -> Any:
        return ref.val

    def setref(self, ref: Ref, val: Any):
        ref.val = val
