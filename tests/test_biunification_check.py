from __future__ import annotations

import dataclasses

import pytest

from some_lang.biunification.type_checker import (
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
