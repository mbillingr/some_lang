import dataclasses

from some_lang.biunification.type_checker import VTypeHead, UTypeHead, Value, Use


@dataclasses.dataclass
class VInt(VTypeHead):
    pass


@dataclasses.dataclass
class UInt(UTypeHead):
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VInt):
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

    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VFunc):
            raise TypeError(self, val)
        return [(val.ret, self.ret), (self.arg, val.arg)]
