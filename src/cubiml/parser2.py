from __future__ import annotations

import abc
import dataclasses
from enum import Enum
from typing import Iterator, TypeVar


class TokenKind(Enum):
    WHITESPACE = 0
    KEYWORD = 1
    IDENTIFIER = 2
    OPERATOR = 3
    LITERAL = 4
    COMMENT = 5


@dataclasses.dataclass(frozen=True)
class Span:
    src: str
    start: int
    end: int


class Node:
    def __init__(self, ch_edges: dict[str, Node] = None, epsilon: set[Node] = None):
        self.ch_edges = ch_edges or {}
        self.epsilon = epsilon or set()

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


class Regex(abc.ABC):
    nfa_start: Node
    nfa_end: Node
    alphabet: set[str]


class Literal(Regex):
    def __init__(self, text: str):
        self.nfa_start = Node()
        current = self.nfa_start

        for ch in text:
            end = Node()
            current.ch_edges[ch] = end
            current = end
        self.nfa_end = end

        self.alphabet = set(text)


class Sequence(Regex):
    def __init__(self, a: Regex, b: Regex):
        a.nfa_end.epsilon.add(b.nfa_start)
        self.nfa_start = a.nfa_start
        self.nfa_end = b.nfa_end
        self.alphabet = a.alphabet | b.alphabet


class Alternative(Regex):
    def __init__(self, *alts: Regex):
        s = Node()
        for a in alts:
            s.epsilon.add(a.nfa_start)

        e = Node()
        for a in alts:
            a.nfa_end.epsilon.add(e)

        self.nfa_start = s
        self.nfa_end = e
        self.alphabet = set()
        for a in alts:
            self.alphabet |= a.alphabet


class Repeat(Regex):
    def __init__(self, x: Regex, accept_empty=True):
        s = Node()
        e = Node()

        if accept_empty:
            s.epsilon.add(e)

        s.epsilon.add(x.nfa_start)
        x.nfa_end.epsilon.add(e)
        x.nfa_end.epsilon.add(x.nfa_start)

        self.nfa_start = s
        self.nfa_end = e
        self.alphabet = x.alphabet


T = TypeVar("T")


class ScannerGenerator:
    def __init__(self):
        self.alphabet: set[str] = set()
        self.nfas: list[Node] = []
        self.accept: dict[Node, T] = {}

    def add_rule(self, token: T, regex: Regex) -> ScannerGenerator:
        self.alphabet |= regex.alphabet
        self.nfas.append(regex.nfa_start)
        self.accept[regex.nfa_end] = token
        return self

    def build(self):
        start = Node()
        for n in self.nfas:
            start.epsilon.add(n)

        q0, subsets, transitions = self._subset_construction(start)

        unvisited = [q0]
        states = {}
        while unvisited:
            subset = unvisited.pop()
            states[subset] = len(states)
            for ch in self.alphabet:
                try:
                    unvisited.append(transitions[subset, ch])
                except KeyError:
                    pass

        state_transitions = {
            (states[a], ch): states[b] for (a, ch), b in transitions.items()
        }

        state_accept = {}
        for subset, st in states.items():
            for node in subset:
                if node not in self.accept:
                    continue
                token = self.accept[node]
                if st in state_accept and state_accept[st] != token:
                    raise RuntimeError(
                        f"Conflicting tokens: {state_accept[st]} and {token}"
                    )
                state_accept[st] = token

        return Scanner(state_accept, state_transitions)

    def _subset_construction(self, start_node: Node):
        transitions = {}
        q0 = epsilon_closure({start_node})
        subsets = {tuple(q0)}
        worklist = [q0]
        while worklist:
            q = worklist.pop()
            for c in self.alphabet:
                t = epsilon_closure(delta(q, c))
                if not t:
                    continue
                transitions[tuple(q), c] = tuple(t)
                if tuple(t) not in subsets:
                    subsets.add(tuple(t))
                    worklist.append(t)
        return tuple(q0), subsets, transitions


def epsilon_closure(nodes: set[Node]) -> set[Node]:
    ec = set()
    while nodes:
        node = nodes.pop()
        if node not in ec:
            ec.add(node)
            nodes |= node.epsilon
    return ec


def delta(nodes: set[Node], ch: str) -> set[Node]:
    out = set()
    for node in nodes:
        try:
            out.add(node.ch_edges[ch])
        except KeyError:
            pass
    return out


class Scanner:
    BAD = object()

    def __init__(self, accept, transitions):
        self._accept = accept
        self._transitions = transitions

    def tokenize(self, src: str) -> Iterator[tuple[str, TokenKind, Span]]:
        start, end = 0, 0
        while start < len(src):
            state = 0
            stack = [self.BAD]
            while True:
                try:
                    ch = src[end]
                except IndexError:
                    break
                if self.accept(state):
                    stack = []
                stack.append(state)
                cat = self.char_cat(ch)
                try:
                    state = self._transitions[state, cat]
                    end += 1
                except KeyError:
                    break

            while not self.accept(state) and state is not self.BAD:
                state = stack.pop()
                end -= 1

            if state is self.BAD:
                raise Exception()
            else:
                yield src[start:end], self._accept[state], Span(src, start, end)

            start = end

    def accept(self, state) -> bool:
        return state in self._accept

    def char_cat(self, ch: str) -> str:
        # could be used to compress the transition table by
        # grouping equal columns into character categories
        return ch


scg = (
    ScannerGenerator()
    .add_rule("FOOBAR", Alternative(Literal("foo"), Literal("bar")))
    .add_rule("FUZZ", Literal("fuzz"))
)

sc = scg.build()
print(list(sc.tokenize("foobarfuzzfoo")))
