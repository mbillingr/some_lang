import dataclasses

import typing

from some_lang import type_heads, ast
from some_lang.biunification.type_checker import (
    TypeCheckerCore,
    VTypeHead,
    UTypeHead,
    Value,
    Use,
)


def test_boolean():
    engine = TypeCheckerCore()
    vbool = engine.get_type(engine.new_val(type_heads.VBool()))
    ubool = engine.get_type(engine.new_use(type_heads.UBool()))
    assert vbool.reify(engine) == ast.BooleanType()
    assert ubool.reify(engine) == ast.BooleanType()


def test_integer():
    engine = TypeCheckerCore()
    vint = engine.new_val(type_heads.VInt())
    uint = engine.new_use(type_heads.UInt())
    assert engine.reify(vint) == ast.IntegerType()
    assert engine.reify(uint) == ast.IntegerType()


def test_vfunction():
    engine = TypeCheckerCore()
    a = engine.new_use(type_heads.UInt())
    r = engine.new_val(type_heads.VInt())
    vfunc = engine.get_type(engine.new_val(type_heads.VFunc(a, r)))
    assert vfunc.reify(engine) == ast.FunctionType(ast.IntegerType(), ast.IntegerType())


def test_ufunction():
    engine = TypeCheckerCore()
    a = engine.new_val(type_heads.VInt())
    r = engine.new_use(type_heads.UInt())
    ufunc = engine.get_type(engine.new_use(type_heads.UFunc(a, r)))
    assert ufunc.reify(engine) == ast.FunctionType(ast.IntegerType(), ast.IntegerType())


def test_higher_order_function():
    engine = TypeCheckerCore()
    v = engine.new_val(type_heads.VInt())
    u = engine.new_use(type_heads.UInt())
    vfunc = engine.new_val(type_heads.VFunc(u, v))
    ufunc = engine.new_use(type_heads.UFunc(v, u))
    vf0 = engine.new_val(type_heads.VFunc(ufunc, vfunc))
    assert engine.reify(vf0) == ast.FunctionType(
        ast.FunctionType(ast.IntegerType(), ast.IntegerType()),
        ast.FunctionType(ast.IntegerType(), ast.IntegerType()),
    )


def test_reify_variable():
    engine = TypeCheckerCore()
    v = engine.new_val(type_heads.VInt())
    u = engine.new_use(type_heads.UInt())
    vv, vu = engine.var()
    engine.flow(v, vu)
    engine.flow(vv, u)
    assert engine.reify_all() == [ast.IntegerType()] * 3


def test_unconstrained_variable():
    engine = TypeCheckerCore()
    engine.var()
    assert engine.reify_all(TVarBuilder()) == [ast.TypeVar("1")]


def test_choose_vtype_for_variables():
    engine = TypeCheckerCore()
    v = engine.new_val(type_heads.VInt())
    u = engine.new_use(UNumber())
    vv, vu = engine.var()
    engine.flow(v, vu)
    engine.flow(vv, u)
    assert engine.reify_all()[-1] == ast.IntegerType()


def test_multiple_vtypes_for_variable():
    engine = TypeCheckerCore()
    v1 = engine.new_val(type_heads.VInt())
    v2 = engine.new_val(VNumber())
    v, u = engine.var()
    engine.flow(v1, u)
    engine.flow(v2, u)
    assert engine.reify_all()[-1] == ast.NumberType()


def test_constrained_variable():
    engine = TypeCheckerCore()
    u = engine.new_use(UNumber())
    v, _ = engine.var()
    engine.flow(v, u)
    assert engine.reify_all()[-1] == "TODO"


@dataclasses.dataclass
class VNumber(VTypeHead):
    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        return ast.NumberType()


@dataclasses.dataclass
class UNumber(UTypeHead):
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        match val:
            case VNumber():
                pass
            case type_heads.VInt():
                pass
            case _:
                raise TypeError(self, val)
        return []

    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        return ast.NumberType()


class TVarBuilder:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return ast.TypeVar(str(self.count))
