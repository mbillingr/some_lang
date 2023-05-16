import dataclasses
import json

import pytest

from eopl_explicit_refs import (
    tokenizer,
    parser,
    interpreter,
    type_checker,
    transform_virtuals,
    rename_qualified,
)
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
        # ((), "[]"),
        ((1, ()), "[1,]"),
        ((1, (2, (3, ()))), "[1, 2, 3]"),
        ((1, (2, (3, ()))), "1 :: [2, 3]"),
        ((1, (2, (3, ()))), "1::2::3::[]"),
        (1, "(the [Int]->Int fn x::xs => x) [1, 2]"),
        ((2, ()), "(the [Int]->[Int] fn x::xs => xs) [1, 2]"),
        ((), "(the [Int]->[Int] fn x::y::ys => ys) [1, 2]"),
        (2, "(the [Int]->Int fn x::y::ys => y) [1, 2]"),
        (0, "let len: [Int]->Int = fn [] => 0 | x::xs => 1 + len xs in len []"),
        (1, "let len: [Int]->Int = fn [] => 0 | x::xs => 1 + len xs in len [0]"),
        (3, "let len: [Int]->Int = fn [] => 0 | x::xs => 1 + len xs in len [0, 0, 0]"),
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
        (0, "let foo: Int -> Int = fn x => x in foo 0"),
        (1, "let zzz: Int -> Int = fn 0 => 1 in zzz 0"),
        (1, "let zzz: Bool -> Int = fn true => 1 in zzz true"),
        (11, "let foo: Int -> Int = fn 0 => 11 | x => x in foo 0"),
        (99, "let foo: Int -> Int = fn 0 => 11 | x => x in foo 99"),
        # Closure
        (1, "(let x = 1 in (the Int -> Int fn y => x)) 0"),
        # Recursive Functions
        (10, "let sum: Int -> Int = fn 0 => 0 | n => n + (sum (n - 1)) in sum 4"),
        (0, "let red: Int -> Int = fn 0 => 0 | n => red (n - 1) in red 10000"),
        # Multiple Arguments & Partial Application
        (6, "(the Int -> Int -> Int -> Int fn a => fn b => fn c => a + b + c) 1 2 3"),
        (5, "((the Int -> Int -> Int fn a => fn b => a + b) 2) 3"),
        (6, "(the Int -> Int -> Int -> Int fn a b c => a + b + c) 1 2 3"),
        (5, "let part = (the Int -> Int -> Int fn a b => a + b) 2 in part 3"),
        (None, "(the () -> () fn () => ()) ()"),
    ],
)
def test_functions(src, expect):
    assert evaluate(src) == expect


@pytest.mark.parametrize(
    "expect, src",
    [
        (0, "let x: Int = 0 in x"),
        (0, "(the Int -> Int fn x => x) 0"),
    ],
)
def test_type_annotations(src, expect):
    assert evaluate(src) == expect


@pytest.mark.parametrize(
    "expect, src",
    [
        # anonymous
        ((1, 2), "[x=1,y=2]"),
        (
            # anonymous record fields sorted alphabetically
            (1, 2),
            "the [x: Int, y: Int] [y = 2, x = 1]",
        ),
        ((1,), "let foo: [x: Int] = [x=1] in foo"),
        ((1,), "(the [x:Int]->[x:Int] fn x => x) [x=1]"),
        # named
        (0, "struct Foo [] 0"),
        (
            # named record fields in declaration order
            (4, 3),
            "struct Foo [y: Int, x: Int] the Foo [x = 3, y = 4]",
        ),
        ((3,), "struct Foo [x: Int] let bar: Foo = [x = 3] in bar"),
        ((3,), "struct Foo [x: Int] (the Foo -> Foo fn x => x) [x = 3]"),
        ((3,), "struct Foo [x: Int] let bar: Foo = (the Foo [x = 3]) in bar"),
        # field access
        (1, "[x=1, y=10, z=100].x"),
        (10, "[y=10, x=1, z=100].y"),
        (100, "[y=10, x=1, z=100].z"),
        (3, "struct Foo [x: Int] let bar: Foo = [x = 3] in bar.x"),
        (0, "struct Foo [x: Int] (the Foo -> Int fn x => x.x) [x = 0]"),
        # struct in struct
        (0, "struct Foo [x: Int] struct Bar [foo: Foo] 0"),
        (0, "[x=[y=[z=0]]].x.y.z"),
        # methods
        (
            0,
            "struct Foo [] impl Foo { method bar: Foo -> () -> Int self () => 0 }" "(the Foo []).bar ()",
        ),
        (
            1,
            "struct Foo [] impl Foo { method bar: Foo -> Int self => 1 }" "(the Foo []).bar",
        ),
        (
            1,
            "struct Foo [] impl Foo { method bar: Self -> Int self => 1 }" "(the Foo []).bar",
        ),
        (
            2,
            "struct Foo [x:Int] impl Foo { method get-x: Foo -> Int self => self.x }" "(the Foo [x=2]).get-x",
        ),
    ],
)
def test_records(src, expect):
    assert evaluate(src) == expect


def test_two_similar_records_are_not_same_type():
    with pytest.raises(TypeError):
        evaluate("struct Foo [x: Int] struct Bar [x: Int] let bar: Bar = (the Foo [x = 3]) in bar")


@pytest.mark.parametrize(
    "expect, src",
    [
        (0, "interface Foo { method bar: Self -> () -> Int } 0"),
        (
            1,
            "interface Foo { method bar: Self -> Int } "
            "struct Bar [] "
            "impl Foo for Bar { method bar: Self -> Int self => 1}"
            "(the Bar []).bar",
        ),
        (
            1,
            "interface Foo { method bar: Self -> Int } "
            "struct Bar [] "
            "impl Foo for Bar { method bar: Self -> Int self => 1}"
            "(the Foo (the Bar [])).bar",
        ),
        (
            (1, (2, ())),
            "interface Foo { method x: Self -> Int } "
            "struct A [] "
            "struct B [] "
            "impl Foo for A { method x: Self -> Int self => 1}"
            "impl Foo for B { method x: Self -> Int self => 2}"
            "let a: A = [] in "
            "let b: B = [] in "
            "let get-x: Foo -> Int = fn obj => obj.x in "
            "    [(get-x a), (get-x b)]",
        ),
        (
            42,
            "interface Foo { method x: Self -> Int } "
            "struct Bar [] "
            "impl Foo for Bar { method x: Self -> Int self => 42 }"
            "struct Fuzz []"
            "impl Fuzz { method y: Self -> Foo self => (the Bar []) }"
            "(the Fuzz []).y.x",
        ),
    ],
)
def test_interfaces(src, expect):
    assert evaluate(src) == expect


def test_impl_of_wrong_type():
    with pytest.raises(TypeError):
        evaluate(
            "interface Foo { method bla: Self -> Bool } "
            "struct Bar [] "
            "impl Foo for Bar { method bla: Self -> Int self => 0 } "
            "0"
        )


def test_missing_method():
    with pytest.raises(TypeError):
        evaluate("interface Foo { method bla: Self -> Self } " "struct Bar [] " "impl Foo for Bar { } " "0")


def test_extra_method():
    with pytest.raises(TypeError):
        evaluate(
            "interface Foo { } " "struct Bar [] " "impl Foo for Bar { method bla: Self -> Self self => self } " "0"
        )


@pytest.mark.parametrize(
    "expect, src",
    [
        (0, "module my-mod { } 0"),
        (0, "module outer { module inner { } } 0"),
        (
            (),
            "module my-mod { struct Foo [] } import .my-mod.Foo (the Foo [])",
        ),
        (
            (),
            "module my-mod { struct Foo [] struct Bar [] } import .my-mod.[Foo Bar] (the Foo [])",
        ),
        (
            0,
            "module my-mod { interface Foo {} } import .my-mod.Foo struct Bar [] impl Foo for Bar {} 0",
        ),
        (
            (),
            "module outer { module inner { struct Foo [] } } import .outer.inner.Foo (the Foo [])",
        ),
        (
            0,
            "module outer { module inner { struct Foo [] } import .inner.Foo } 0",
        ),
        (
            0,
            "module outer { module inner { struct Foo [] } import __main__.outer.inner.Foo } 0",
        ),
        (
            1,
            "module my-mod { "
            "    struct Bar [] "
            "    impl Bar { method x: Self -> Int self => 1 }"
            "}"
            "import .my-mod.Bar (the Bar []).x",
        ),
        (
            2,
            "module my-mod { "
            "    interface Foo { method x: Self -> Int } "
            "    struct Bar [] "
            "    impl Foo for Bar { method x: Self -> Int self => 2 }"
            "}"
            "import .my-mod.[Foo Bar] (the Foo (the Bar [])).x",
        ),
        (
            42,
            "module my-mod { "
            "    interface Foo { method x: Self -> Int } "
            "    struct Bar [] "
            "    impl Foo for Bar { method x: Self -> Int self => 42 }"
            "    struct Fuzz []"
            "    impl Fuzz { method y: Self -> Foo self => (the Bar []) }"
            "}"
            "import .my-mod.[Foo Fuzz] (the Fuzz []).y.x",
        ),
    ],
)
def test_modules(src, expect):
    assert evaluate(src) == expect


def test_module_scoping():
    with pytest.raises(LookupError):
        evaluate("module outer { module inner { struct Foo [] } } (the Foo [])")


def evaluate(src):
    token_stream = tokenizer.default_tokenizer(src)
    program = parser.parse_program(token_stream)
    program = rename_qualified.rename_qualified(program)
    print(json.dumps(program.to_dict(), indent=4))
    checked = type_checker.check_program(program)
    checked = transform_virtuals.transform_virtuals(checked)
    # print(json.dumps(checked.to_dict(), indent=4))
    runner = interpreter.analyze_program(checked)
    return runner(Store())
