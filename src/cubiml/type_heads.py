import dataclasses

from biunification.type_checker import VTypeHead, UTypeHead, Value, Use


@dataclasses.dataclass
class VBool(VTypeHead):
    pass


@dataclasses.dataclass
class UBool(UTypeHead):
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VBool):
            raise TypeError(self, val)
        return []


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
class VObj(VTypeHead):
    fields: dict[str, Value]


@dataclasses.dataclass
class UObj(UTypeHead):
    field: str
    use: Use

    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VObj):
            raise TypeError(self, val)
        try:
            return [(val.fields[self.field], self.use)]
        except KeyError:
            raise TypeError("Missing Field", self.field) from None


@dataclasses.dataclass
class VCase(VTypeHead):
    tag: str
    typ: Value


@dataclasses.dataclass
class UCase(UTypeHead):
    cases: dict[str, Use]

    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VCase):
            raise TypeError(self, val)
        try:
            return [(val.typ, self.cases[val.tag])]
        except KeyError:
            raise TypeError("Unhandled Case", val.tag) from None


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
