from __future__ import annotations
import abc
import dataclasses
from typing import TypeVar, Optional, Generic

T = TypeVar("T")


class Env(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def apply(self, identifier: str) -> Optional[T]:
        """Apply the environment to an identifier. This could be o simple lookup,
        or something more involved, like initializing a recursive function."""

    def extend(self, identifier: str, value: T) -> Env:
        """Extend the environment with a binding"""
        return EnvEntry(identifier, value, self)


class EmptyEnv(Env):
    def apply(self, identifier: str) -> Optional[T]:
        return None


@dataclasses.dataclass
class EnvEntry(Env, Generic[T]):
    ident: str
    value: T
    saved_env: Env

    def apply(self, identifier: str) -> Optional[T]:
        if self.ident == identifier:
            return self.value
        return self.saved_env.apply(identifier)
