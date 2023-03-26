import dataclasses
from typing import Any


@dataclasses.dataclass
class Ref:
    r: Any


class PythonStore:
    def clear(self):
        pass

    def is_reference(self, x: Any) -> bool:
        return isinstance(x, Ref)

    def newref(self, val: Any) -> Ref:
        return Ref(val)

    def deref(self, ref: Ref) -> Any:
        return ref.r

    def setref(self, ref: Ref, val: Any):
        ref.r = val
