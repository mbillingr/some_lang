from __future__ import annotations

import abc
import bisect
import dataclasses
import json
from typing import Iterator, TypeVar, Iterable, Any, Generic, Self


class ScannerError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Span:
    src: str
    start: int
    end: int

    @classmethod
    def make_eof(cls, src) -> Self:
        return cls(src, len(src) - 1, len(src) + 1)

    @classmethod
    def virtual(cls) -> Self:
        return cls("", -1, -1)

    def is_virtual(self) -> bool:
        return self.src == "" and self.start == -1 and self.end == -1

    def merge(self, other: Span) -> Span:
        if self.is_virtual():
            return other
        if other.is_virtual():
            return self
        return Span(self.src, min(self.start, other.start), max(self.end, other.end))

    def show_line(self, marker="^", n_before=1) -> str:
        newlines = [-1, *self._find_newlines(), len(self.src)]
        start_line = bisect.bisect_left(newlines, self.start)
        line_offset = 1 + newlines[start_line - 1]
        highlight_offset = self.start - line_offset
        highlight_len = min(self.end, newlines[start_line]) - self.start
        output = []
        for line in range(max(1, start_line - n_before), start_line + 1):
            row = f"{line:>5}: {self.src[1+newlines[line-1]:newlines[line]]}"
            output.append(row)
        output.append(
            " " * (5 + 2 + highlight_offset)
            + marker * highlight_len
            + ("↵" if self.end > newlines[start_line] else "")
        )
        return "\n".join(output)

    def _find_newlines(self):
        ofs = -1
        while True:
            ofs = self.src.find("\n", ofs + 1)
            if ofs == -1:
                return
            yield ofs


class Regex(abc.ABC):
    """A regular expression, used for constructing scanners"""

    nfa_start: Node
    nfa_end: Node
    alphabet: set[str]

    def __add__(self, other):
        return Sequence(self, to_regex(other))

    def __radd__(self, other):
        return Sequence(to_regex(other), self)

    def __or__(self, other):
        return Alternative(self, to_regex(other))

    def __ror__(self, other):
        return Alternative(to_regex(other), self)


def to_regex(r: Any) -> Regex:
    match r:
        case Regex():
            return r
        case str():
            return Literal(r)
        case _:
            raise TypeError("Invalid Regex", r)


class Literal(Regex):
    """Matches a literal string"""

    def __init__(self, text: str):
        self.nfa_start = Node()
        end = self.nfa_start
        current = self.nfa_start

        for ch in text:
            end = Node()
            current.ch_edges[ch] = end
            current = end
        self.nfa_end = end

        self.alphabet = set(text)


class Sequence(Regex):
    """Matches other regular expressions in sequence"""

    def __init__(self, first, *rest):
        first = to_regex(first)
        rest = tuple(map(to_regex, rest))

        self.nfa_start = first.nfa_start
        self.nfa_end = first.nfa_end
        self.alphabet = first.alphabet

        for x in rest:
            self.nfa_end.epsilon.add(x.nfa_start)
            self.nfa_end = x.nfa_end
            self.alphabet |= x.alphabet


class Alternative(Regex):
    """Matches any one of the given regular expressions"""

    def __init__(self, *alts):
        alts = tuple(map(to_regex, alts))

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


class OneOf(Alternative):
    """Matches any one of the given strings/chars"""

    def __init__(self, alts: Iterable[str]):
        super().__init__(*map(Literal, alts))


class Repeat(Regex):
    """Matches repetitions of a regular expression"""

    def __init__(self, x, accept_empty=True):
        x = to_regex(x)

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


class Opt(Regex):
    """Optionally matches the given regular expressions"""

    def __init__(self, x):
        x = to_regex(x)

        s = Node()
        e = Node()

        s.epsilon.add(e)

        s.epsilon.add(x.nfa_start)
        x.nfa_end.epsilon.add(e)

        self.nfa_start = s
        self.nfa_end = e
        self.alphabet = x.alphabet


T = TypeVar("T")


class ScannerGenerator(Generic[T]):
    """Generate a scanner from pairs of tokens and regexes"""

    def __init__(self):
        self.alphabet: set[str] = set()
        self.nfas: list[Node] = []
        self.accept: dict[Node, T] = {}
        self.token_preference: dict[tuple[T, T], T] = {}

    def set_token_priority(self, select: T, discard: T) -> ScannerGenerator:
        worklist = [(select, discard)]

        for (a, b), c in self.token_preference.items():
            if a == c:
                if discard == a:
                    worklist.append((select, b))
                if select == b:
                    worklist.append((a, discard))
            else:
                if discard == b:
                    worklist.append((select, a))
                if select == a:
                    worklist.append((b, discard))

        for s, d in worklist:
            self.token_preference[(s, d)] = s
            self.token_preference[(d, s)] = s

        return self

    def add_rule(self, token: T, regex: Any) -> ScannerGenerator:
        regex = to_regex(regex)
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
            if subset in states:
                continue
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
                    token = self.token_preference.get((token, state_accept[st]))
                if not token:
                    raise RuntimeError(
                        f"Conflicting tokens: {state_accept[st]} and {self.accept[node]}"
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


class Node:
    def __init__(self, ch_edges: dict[str, Node] = None, epsilon: set[Node] = None):
        self.ch_edges = ch_edges or {}
        self.epsilon = epsilon or set()

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


def epsilon_closure(nodes: set[Node]) -> set[Node]:
    """extend a set of nodes with all nodes reachable through epsilon edges"""
    ec = set()
    while nodes:
        node = nodes.pop()
        if node not in ec:
            ec.add(node)
            nodes |= node.epsilon
    return ec


def delta(nodes: set[Node], ch: str) -> set[Node]:
    """compute the transition between two sets of nodes"""
    out = set()
    for node in nodes:
        try:
            out.add(node.ch_edges[ch])
        except KeyError:
            pass
    return out


class Scanner(Generic[T]):
    """A lexical scanner."""

    BAD = object()
    ERR = object()

    def __init__(self, accept, transitions):
        self._accept = accept
        self._transitions = transitions

    def tokenize(self, src: str) -> Iterator[tuple[str, T, Span]]:
        """Split an input string into tokens and iterate over them."""
        start, end = 0, 0
        while start < len(src):
            state = 0
            stack = [self.BAD]
            while True:
                if self.accept(state):
                    stack = []
                stack.append(state)

                try:
                    ch = src[end]
                except IndexError:
                    break
                cat = self.char_cat(ch)

                try:
                    state = self._transitions[state, cat]
                except KeyError:
                    state = self.ERR
                    break

                end += 1

            furthest = end

            if not self.accept(state):
                state = stack.pop()

            while not self.accept(state) and state is not self.BAD:
                state = stack.pop()
                end -= 1

            if state is self.BAD:
                raise ScannerError(
                    f"`{src[start:furthest]}` followed by `{src[furthest:furthest+1]}`"
                )
            else:
                yield src[start:end], self._accept[state], Span(src, start, end)

            start = end

    def accept(self, state) -> bool:
        return state in self._accept

    def char_cat(self, ch: str) -> str:
        # could be used to compress the transition table by
        # grouping equal columns into character categories
        return ch

    def store(self, filename):
        transitions = {}
        for k, v in self._transitions.items():
            transitions.setdefault(k[0], {})[k[1]] = v
        with open(filename, "w") as f:
            json.dump({"accept": self._accept, "transitions": transitions}, f, indent=4, sort_keys=True)

    @staticmethod
    def load(TokenType, filename):
        with open(filename) as f:
            data = json.load(f)

        transitions = {}
        for state, edge in data["transitions"].items():
            for ch, newstate in edge.items():
                transitions[(int(state), ch)] = newstate

        accept = {int(k): TokenType(v) for k, v in data["accept"].items()}

        return Scanner(accept, transitions)


def num():
    return OneOf("0123456789")


scg = (
    ScannerGenerator()
    .add_rule("FOOBAR", Alternative("foo", "bar"))
    .add_rule("FUZZ", "fuzz")
    .add_rule(
        "NUM",
        Repeat(num(), accept_empty=False)
        + Opt("." + Repeat(num(), accept_empty=False)),
    )
    .add_rule("WHITESPACE", Repeat(OneOf(" \t"), accept_empty=False))
    .add_rule("RIPRAP", "a" + Repeat("bc"))
)
