from typing import Any, Optional

from unification.substitution import Substitution, Var
from unification.structure import structural_visitor


class NoOccurrenceViolation(Exception):
    pass


class UnificationFailure(Exception):
    pass


def unify(
    a: Any, b: Any, subst: Substitution, context: Optional[Any] = None
) -> Substitution:
    a = subst.apply(a)
    b = subst.apply(b)

    if a == b:
        return subst

    if isinstance(a, Var):
        if occurs(a, b):
            raise NoOccurrenceViolation(a, b, context)
        else:
            return subst.extend(a, b)

    if isinstance(b, Var):
        if occurs(b, a):
            raise NoOccurrenceViolation(b, a, context)
        else:
            return subst.extend(b, a)

    if type(a) != type(b):
        raise UnificationFailure(a, b, context)

    ai = structural_visitor(a, lambda x: x, lambda x: x)
    bi = structural_visitor(b, lambda x: x, lambda x: x)
    for a_, b_ in zip(ai, bi):
        subst = unify(a_, b_, subst, context)

    if has_remaining_items(ai) or has_remaining_items(bi):
        raise UnificationFailure(a, b, context)

    return subst


def occurs(v: Var, struc: Any) -> bool:
    match struc:
        case Var():
            return struc is v
        case _:
            return structural_visitor(struc, lambda x: occurs(v, x), reducer=any)


def has_remaining_items(it) -> bool:
    try:
        next(it)
    except StopIteration:
        return False
    return True
