import abc
import dataclasses


class Expression(abc.ABC):
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


TRUE = Boolean(True)
FALSE = Boolean(False)
