import abc
import dataclasses


class Expr(abc.ABC):
    pass


@dataclasses.dataclass
class Integer(Expr):
    val: int


@dataclasses.dataclass
class Reference(Expr):
    var: str


@dataclasses.dataclass
class Lambda(Expr):
    var: str
    bdy: Expr


@dataclasses.dataclass
class Application(Expr):
    rator: Expr
    rand: Expr
