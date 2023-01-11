from functools import singledispatch
from typing import Any, Callable


@singledispatch
def structural_visitor(struc: Any, visitor: Callable[[Any], Any], reducer=None) -> Any:
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


@structural_visitor.register
def _(struc: list, visitor: Callable[[Any], Any], reducer=list) -> Any:
    return reducer(visitor(x) for x in struc)
