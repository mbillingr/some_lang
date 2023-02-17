from __future__ import annotations

import abc
import dataclasses
import functools
from enum import Enum
from typing import Iterator, Iterable


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


EOF = object()

ACCEPT = {
    0: False,
    1: True,
}

KIND = {1: TokenKind.WHITESPACE}

TRANSITION = {
    (0, "WHITESPACE"): 1,
    (1, "WHITESPACE"): 1,
}


def char_cat(ch: str) -> str:
    if ch is EOF:
        return "EOF"

    if ch.isspace():
        return "WHITESPACE"

    raise ValueError(ch)


def tokenize(src: str) -> Iterator[tuple[str, TokenKind, Span]]:
    state = 0
    start, end = 0, 0
    BAD = object()
    while start < len(src):
        stack = [BAD]
        while True:
            try:
                ch = src[end]
            except IndexError:
                break
            end += 1
            if ACCEPT[state]:
                stack = []
            stack.append(state)
            cat = char_cat(ch)
            state = TRANSITION[state, cat]

        while not ACCEPT[state] and state is not BAD:
            state = stack.pop()
            end -= 1

        if state is BAD:
            raise Exception()
        else:
            yield src[start:end], KIND[state], Span(src, start, end)

        start = end


print(list(tokenize("   ")))


class Node:
    def __init__(self, ch_edges: dict[str, Node] = None, epsilon: set[Node] = None):
        self.ch_edges = ch_edges or {}
        self.epsilon = epsilon or set()

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


# constructing a nondeterministic finite automaton


def char(ch):
    a = Node()
    b = Node()
    a.ch_edges[ch] = b
    return [a, b]


def seq(a, b):
    a[-1].epsilon.add(b[0])
    return a + b


def alt(a, b):
    s = Node()
    s.epsilon.add(a[0])
    s.epsilon.add(b[0])

    e = Node()
    a[-1].epsilon.add(e)
    b[-1].epsilon.add(e)

    return [s, *a, *b, e]


def rep(x, accept_zero=True):
    s = Node()
    e = Node()
    if accept_zero:
        s.epsilon.add(e)
    s.epsilon.add(x[0])
    x[-1].epsilon.add(e)
    x[-1].epsilon.add(x[0])
    return [s, *x, e]


def show(nfa):
    for a in nfa:
        for b in a.epsilon:
            print(id(a), "----->", id(b))
        for ch, c in a.ch_edges.items():
            print(id(a), f"--{ch}-->", id(c))
        if not a.epsilon and not a.ch_edges:
            print(id(a))


nfa = seq(char("A"), rep(alt(char("B"), char("C"))))
show(nfa)


# NFA to DFA


class NodeSet(set):
    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


def subset_construction(nfa):
    transitions = {}
    q0 = epsilon_closure({nfa[0]})
    Q = {tuple(q0)}
    worklist = [q0]
    while worklist:
        q = worklist.pop()
        for c in ALPHABET:
            t = epsilon_closure(delta(q, c))
            if not t:
                continue
            transitions[tuple(q), c] = tuple(t)
            if tuple(t) not in Q:
                Q.add(tuple(t))
                worklist.append(t)
    return transitions


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


ALPHABET = "ABC"


def show_transitions(tra):
    for (a, ch), b in tra.items():
        print(hash(a), f"--{ch}-->", hash(b))


print("-----")
show_transitions(subset_construction(nfa))


def open_alt(a, b):
    s = Node()
    s.epsilon.add(a[0])
    s.epsilon.add(b[0])

    return [s, *a, *b]


nfa = open_alt(seq(char("A"), char("B")), seq(char("A"), char("C")))
print("======")
show(nfa)
print("-----")
show_transitions(subset_construction(nfa))


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
