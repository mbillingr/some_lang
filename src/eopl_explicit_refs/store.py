import dataclasses
from typing import Any


class PythonStore:
    @dataclasses.dataclass
    class Ref:
        val: Any

    def __init__(self):
        self.env = ()

    def clear(self):
        self.env = ()

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
