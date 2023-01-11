from functools import singledispatch
from typing import Any, Callable, Optional, Self


class Var:
    pass


class Substitution:
    def __init__(self, subs: Optional[dict[Var, Any]] = None):
        self.subs = subs or {}

    def extend(self, var: Var, struc: Any) -> Self:
        new_subs = {v: apply_one_subst(s, var, struc) for v, s in self.subs.items()}
        new_subs[var] = struc
        return Substitution(new_subs)

    def apply(self, struc: Any) -> Any:
        match struc:
            case Var():
                return self.subs.get(struc, struc)
            case _:
                return structural_visit(struc, self.apply)


def apply_one_subst(s0: Any, var: Var, s1: Any) -> Any:
    match s0:
        case Var():
            return s1 if s0 is var else s0
        case _:
            return structural_visit(s0, lambda x: apply_one_subst(x, var, s1))


@singledispatch
def structural_visit(struc: Any, visitor: Callable[[Any], Any], reducer=None) -> Any:
    """Any type that can be unified should register to this function. Its expected behavior is a bit tricky...

    The implementation should call the visitor on all the value's child elements.

    If the caller provides no value for the reducer argument, the implementation is expected to construct a new
    value of its type from the results of the visitor calls.

    However, if a reducer is provided, the implementation should return the result of passing an iterator over the
    results of the visitor calls to the reducer.
    """
    if reducer is None:
        return struc
    return reducer(_ for _ in ())


@structural_visit.register
def _(struc: list, visitor: Callable[[Any], Any], reducer=list) -> Any:
    return reducer(visitor(x) for x in struc)
