from __future__ import annotations
from typing import Any, Optional, Self

from some_lang.unification.structure import structural_visitor


class Var:
    pass


class Substitution:
    def __init__(self, subs: Optional[dict[Var, Any]] = None):
        self.subs = subs or {}

    def extend(self, var: Var, struc: Any) -> Substitution:
        new_subs = {v: apply_one_subst(s, var, struc) for v, s in self.subs.items()}
        new_subs[var] = struc
        return Substitution(new_subs)

    def apply(self, struc: Any) -> Any:
        match struc:
            case Var():
                return self.subs.get(struc, struc)
            case _:
                return structural_visitor(struc, self.apply)


def apply_one_subst(s0: Any, var: Var, s1: Any) -> Any:
    match s0:
        case Var():
            return s1 if s0 is var else s0
        case _:
            return structural_visitor(s0, lambda x: apply_one_subst(x, var, s1))
