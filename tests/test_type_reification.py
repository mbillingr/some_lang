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


def test_choose_vtype_for_variables():
    engine = TypeCheckerCore()
    v = engine.new_val(type_heads.VInt())
    u = engine.new_use(UNumber())
    vv, vu = engine.var()
    engine.flow(v, vu)
    engine.flow(vv, u)
    assert engine.reify_all() == [ast.IntegerType(), "Number", ast.IntegerType()]


def test_use_most_specific_type():
    engine = TypeCheckerCore()
    i = engine.new_val(type_heads.VInt())
    v = engine.new_val(VNumber())
    u = engine.new_use(UNumber())
    engine.flow(v, u)
    engine.flow(i, u)
    print(engine)
    assert engine.reify_all() == [ast.IntegerType(), "Number", ast.IntegerType()]


@dataclasses.dataclass
class VNumber(VTypeHead):
    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        return "Number"


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
        return "Number"
