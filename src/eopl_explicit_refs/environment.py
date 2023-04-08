from __future__ import annotations
import abc


class Env(abc.ABC):
    def extend(self, *vars: str):
        return Level(vars, self)

    @abc.abstractmethod
    def lookup(self, var: str) -> int:
        pass


class EmptyEnv(Env):
    def __init__(self):
        pass

    def lookup(self, var: str) -> int:
        raise LookupError(var)


class Level(Env):
    def __init__(self, vars: tuple[str, ...], nxt: Env):
        self.vars = vars
        self.nxt = nxt

    def lookup(self, var: str) -> tuple[int, int]:
        try:
            return self.vars.index(var), 0
        except ValueError:
            pass
        ofs, depth = self.nxt.lookup(var)
        return ofs, depth + 1
