import dataclasses

from some_lang.biunification.type_checker import (
    VTypeHead,
    UTypeHead,
    Value,
    Use,
    TypeCheckerCore,
)


@dataclasses.dataclass
class VBool(VTypeHead):
    def get_use(self) -> Use:
        return UBool()
    def substitute(self, t: int, v: Value, u: Use):
        return self


@dataclasses.dataclass
class UBool(UTypeHead):
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VBool):
            raise TypeError(self, val)
        return []

    def substitute(self, t: int, v: Value, u: Use):
        return self


@dataclasses.dataclass
class VInt(VTypeHead):
    def get_use(self) -> Use:
        return UInt()
    def substitute(self, t: int, v: Value, u: Use):
        return self


@dataclasses.dataclass
class UInt(UTypeHead):
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VInt):
            raise TypeError(self, val)
        return []

    def substitute(self, t: int, v: Value, u: Use):
        return self


@dataclasses.dataclass
class VFunc(VTypeHead):
    arg: Use
    ret: Value

    def get_use(self) -> Use:
        raise NotImplementedError()

    def substitute(self, t: int, v: Value, u: Use):
        arg = u if self.arg == t else self.arg
        ret = v if self.ret == t else self.ret
        return VFunc(arg, ret)


@dataclasses.dataclass
class UFunc(UTypeHead):
    arg: Value
    ret: Use

    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VFunc):
            raise TypeError(self, val)
        return [(val.ret, self.ret), (self.arg, val.arg)]

    def substitute(self, t: int, v: Value, u: Use):
        arg = v if self.arg == t else self.arg
        ret = u if self.ret == t else self.ret
        return UFunc(arg, ret)
