from __future__ import annotations

import dataclasses
from copy import deepcopy

import pytest

from biunification.type_checker import (
    TypeCheckerCore,
    VTypeHead,
    UTypeHead,
    Value,
    Use,
)


def test_bool_pass():
    tc = TypeCheckerCore()
    v = tc.new_val(VBool())
    u = tc.new_use(UBool())

    tc.flow(v, u)  # should not raise


def test_type_error():
    tc = TypeCheckerCore()
    v = tc.new_val(VFunc(tc.new_use(UBool()), tc.new_val(VBool())))
    u = tc.new_use(UBool())

    with pytest.raises(TypeError):
        tc.flow(v, u)


def test_type_pass_through_var():
    tc = TypeCheckerCore()
    v, u = tc.var()
    bv = tc.new_val(VBool())
    bu = tc.new_use(UBool())

    tc.flow(v, bu)
    tc.flow(bv, u)
    # should not raise


def test_type_fail_through_var():
    tc = TypeCheckerCore()
    v, u = tc.var()
    b = tc.new_use(UBool())
    f = tc.new_val(VFunc(tc.new_use(UBool()), tc.new_val(VBool())))

    with pytest.raises(TypeError):
        tc.flow(v, b)
        tc.flow(f, u)


def test_no_cycles_to_collapse():
    tc = TypeCheckerCore()
    v1, u1 = tc.var()
    v2, u2 = tc.var()
    tc.flow(v1, u2)

    original = deepcopy(tc)
    tc.collapse_cycles()

    assert tc.types == original.types
    assert tc.r == original.r


def test_collapse_simple_cycle():
    tc = TypeCheckerCore()
    v0, _ = tc.var()
    v1, u1 = tc.var()
    v3, u2 = tc.var()
    tc.flow(v0, u1)
    tc.flow(v1, u2)
    tc.flow(v3, u1)

    tc.collapse_cycles()

    assert tc.types == ["Var", "Var", "erased"]
    assert tc.r.downsets[0] == {1}
    assert tc.r.upsets[0] == set()
    assert tc.r.downsets[1] == set()
    assert tc.r.upsets[1] == {0}


def test_collapse_self_cycle():
    tc = TypeCheckerCore()
    v1, u1 = tc.var()
    tc.flow(v1, u1)

    tc.collapse_cycles()

    assert tc.types == ["Var"]
    assert tc.r.downsets[0] == set()
    assert tc.r.upsets[0] == set()


@dataclasses.dataclass
class VBool(VTypeHead):
    pass


@dataclasses.dataclass
class UBool(UTypeHead):
    def check(self, val: VTypeHead) -> list[(Value, Use)]:
        if not isinstance(val, VBool):
            raise TypeError(self, val)
        return []


@dataclasses.dataclass
class VFunc(VTypeHead):
    arg: Use
    ret: Value


@dataclasses.dataclass
class UFunc(UTypeHead):
    arg: Value
    ret: Use

    def check(self, val: VTypeHead) -> list[(Value, Use)]:
        if not isinstance(val, VFunc):
            raise TypeError(self, val)
        return [(self.ret, val.ret), (val.arg, self.arg)]
