import dataclasses

import typing

from some_lang import ast
from some_lang.biunification.type_checker import (
    VTypeHead,
    UTypeHead,
    Value,
    Use,
    TypeCheckerCore,
)


@dataclasses.dataclass
class VBool(VTypeHead):
    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        return ast.BooleanType()


@dataclasses.dataclass
class UBool(UTypeHead):
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VBool):
            raise TypeError(self, val)
        return []

    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        return ast.BooleanType()


@dataclasses.dataclass
class VInt(VTypeHead):
    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        return ast.IntegerType()


@dataclasses.dataclass
class UInt(UTypeHead):
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VInt):
            raise TypeError(self, val)
        return []

    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        return ast.IntegerType()


@dataclasses.dataclass
class VFunc(VTypeHead):
    arg: Use
    ret: Value

    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        return ast.FunctionType(engine.reify(self.arg), engine.reify(self.ret))


@dataclasses.dataclass
class UFunc(UTypeHead):
    arg: Value
    ret: Use

    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VFunc):
            raise TypeError(self, val)
        return [(val.ret, self.ret), (self.arg, val.arg)]

    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        return ast.FunctionType(engine.reify(self.arg), engine.reify(self.ret))
