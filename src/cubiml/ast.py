import abc
import dataclasses


class ToplevelItem(abc.ABC):
    pass


class Expression(ToplevelItem):
    pass


@dataclasses.dataclass(frozen=True)
class Boolean(Expression):
    val: bool


@dataclasses.dataclass(frozen=True)
class Reference(Expression):
    var: str


@dataclasses.dataclass(frozen=True)
class Conditional(Expression):
    condition: Expression
    consequence: Expression
    alternative: Expression


@dataclasses.dataclass(frozen=True)
class Record(Expression):
    fields: dict[str, Expression]


@dataclasses.dataclass(frozen=True)
class FieldAccess(Expression):
    field: str
    expr: Expression


@dataclasses.dataclass(frozen=True)
class Function(Expression):
    var: str
    body: Expression


@dataclasses.dataclass(frozen=True)
class Application(Expression):
    fun: Expression
    arg: Expression


@dataclasses.dataclass(frozen=True)
class Let(Expression):
    var: str
    val: Expression
    body: Expression


@dataclasses.dataclass(frozen=True)
class FuncDef(Expression):
    var: str
    fun: Function


@dataclasses.dataclass(frozen=True)
class LetRec(Expression):
    bind: list[FuncDef]
    body: Expression


@dataclasses.dataclass(frozen=True)
class DefineLet(ToplevelItem):
    var: str
    val: Expression


@dataclasses.dataclass(frozen=True)
class DefineLetRec(ToplevelItem):
    bind: list[FuncDef]


@dataclasses.dataclass(frozen=True)
class Script:
    statements: list[ToplevelItem]


TRUE = Boolean(True)
FALSE = Boolean(False)
