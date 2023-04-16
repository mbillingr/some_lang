import pytest

from eopl_explicit_refs import tokenizer, parser, interpreter
from eopl_explicit_refs.store import PythonStore as Store


@pytest.mark.parametrize(
    "expect, src",
    [
        (False, "false"),
        (True, "true"),
        (0, "0"),
        (42, "42"),
        (-123, "-123"),
    ],
)
def test_literals(src, expect):
    assert evaluate(src) == expect


@pytest.mark.parametrize(
    "expect, src",
    [
        # Operators
        (3, "1 + 2"),
        (-1, "1 - 2"),
        (12, "6 * 2"),
        (3, "6 / 2"),
        (11, "1 + 2 * 3 + 4"),
        (21, "(1 + 2) * (3 + 4)"),
    ],
)
def test_operators(src, expect):
    assert evaluate(src) == expect


@pytest.mark.parametrize(
    "expect, src",
    [
        # Lists
        ((), "[]"),
        ((1, (2, (3, ()))), "[1 2 3]"),
        ((1, (2, (3, ()))), "1 :: [2 3]"),
        ((1, (2, (3, ()))), "1::2::3::[]"),
        (1, "(fn x::xs => x) [1 2]"),
        ((2, ()), "(fn x::xs => xs) [1 2]"),
        ((), "(fn x::y::ys => ys) [1 2]"),
        (2, "(fn x::y::ys => y) [1 2]"),
        (0, "let len = fn [] => 0 | x::xs => 1 + len xs in len []"),
        (1, "let len = fn [] => 0 | x::xs => 1 + len xs in len [0]"),
        (3, "let len = fn [] => 0 | x::xs => 1 + len xs in len [0 0 0]"),
    ],
)
def test_lists(src, expect):
    assert evaluate(src) == expect


@pytest.mark.parametrize(
    "expect, src",
    [
        # Sequence
        (0, "{ 0 }"),
        (3, "{ 1; 2; 3 }"),
    ],
)
def test_sequence(src, expect):
    assert evaluate(src) == expect


@pytest.mark.parametrize(
    "expect, src",
    [
        # Conditional Expression
        (1, "if true then 1 else 2"),
        (2, "if false then 1 else 2"),
        # If Statement
        (1, "let x = newref 0 in {if true then set x = 1 else set x = 2; deref x}"),
        (2, "let x = newref 0 in {if false then set x = 1 else set x = 2; deref x}"),
        (1, "let x = newref 0 in {if true then set x = 1; deref x}"),
        (0, "let x = newref 0 in {if false then set x = 1; deref x}"),
    ],
)
def test_conditional(src, expect):
    assert evaluate(src) == expect


@pytest.mark.parametrize(
    "expect, src",
    [
        # Binding
        (0, "let x = 0 in x"),
        (1, "let x = 0 in let x = 1 in x"),
        (0, "let x = 0 in let y = 1 in x"),
        (0, "let x = 0 in {1; 2; x}"),
        # References
        (Store.Ref(1), "newref 1"),
        (2, "let x = newref 2 in deref x"),
        (42, "let x = newref 0 in {set x = 42; deref x}"),
    ],
)
def test_bindings(src, expect):
    assert evaluate(src) == expect


@pytest.mark.parametrize(
    "expect, src",
    [
        # Anonymous Functions
        (0, "let foo = fn x => x in foo 0"),
        (1, "let zzz = fn 0 => 1 in zzz 0"),
        (1, "let zzz = fn true => 1 in zzz true"),
        (11, "let foo = fn 0 => 11 | x => x in foo 0"),
        (99, "let foo = fn 0 => 11 | x => x in foo 99"),
        # Closure
        (1, "(let x = 1 in (fn y => x)) 0"),
        # Recursive Functions
        (10, "let sum = fn 0 => 0 | n => n + (sum (n - 1)) in sum 4"),
        (0, "let red = fn 0 => 0 | n => red (n - 1) in red 10000"),
        # Multiple Arguments & Partial Application
        (6, "(fn a => fn b => fn c => a + b + c) 1 2 3"),
        (6, "(((fn a => fn b => fn c => a + b + c) 1) 2) 3"),
        (6, "(fn a b c => a + b + c) 1 2 3"),
        (6, "let part = (fn a b c => a + b + c) 1 2 in part 3"),
    ],
)
def test_functions(src, expect):
    assert evaluate(src) == expect


@pytest.mark.parametrize(
    "expect, src",
    [
        # Classes
        (0, "class Foo { } let x = new Foo in 0"),
        (0, "class Foo { method init x y z => 123 } let x = new Foo 1 2 3 in 0"),
        (0, "class Foo { method bar x => x } let foo = new Foo in send foo Foo bar 0"),
        (0, "class Foo { method bar => 0 } let foo = new Foo in send foo Foo bar"),
        (1, "class A { method foo => 1 } class B extends A { } let x = new B in send x B foo"),
        (2, "class A { method foo => 1 } class B extends A { method foo => 2 } let x = new B in send x A foo"),
        (0, "class Foo { field f } let x = new Foo in 0"),
        (
            0,
            "class Foo { field f method init => {setfield f = 0} method get => getfield f }"
            "let x = new Foo in send x Foo get",
        ),
    ],
)
def test_classes(src, expect):
    assert evaluate(src) == expect


def evaluate(src):
    token_stream = tokenizer.default_tokenizer(src)
    program = parser.parse_program(token_stream)
    runner = interpreter.analyze_program(program)
    return runner(Store())
