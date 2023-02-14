import pytest

from cubiml import parser, type_checker, gen_cpp, gen_rust, gen_rust2, interpreter


def eval_in_rust(src: str) -> str:
    ast = parser.parse_script(src)
    tck = type_checker.TypeChecker()
    typemap = tck.check_script(ast)
    runner = gen_rust.Runner(typemap, tck.engine)
    return runner.run_script(ast)


def eval_in_rust2(src: str) -> str:
    ast = parser.parse_script(src)
    tck = type_checker.TypeChecker()
    typemap = tck.check_script(ast)
    runner = gen_rust2.Runner(typemap, tck.engine)
    return runner.run_script(ast)


def eval_in_cpp(src: str) -> str:
    ast = parser.parse_script(src)
    tck = type_checker.TypeChecker()
    typemap = tck.check_script(ast)
    runner = gen_cpp.Runner(typemap, tck.engine)
    return runner.run_script(ast)


def eval_in_python(src: str) -> str:
    ast = parser.parse_script(src)
    type_checker.TypeChecker().check_script(ast)
    runner = interpreter.Interpreter()
    res = runner.run_script(ast)
    return transform_python_result(res)


@pytest.mark.parametrize("evaluator", [eval_in_python, eval_in_rust2])
class TestLanguage:
    def test_just_a_literal(self, evaluator):
        src = "42"
        res = evaluator(src)
        assert res == "42"

    def test_operators(self, evaluator):
        src = "10 - 2 * (2 + 3) == 0"
        res = evaluator(src)
        assert res == "true"

    def test_toplevel_binding(self, evaluator):
        src = "let x = false; x"
        res = evaluator(src)
        assert res == "false"

    def test_function_application(self, evaluator):
        src = "(fun x -> x) {}"
        res = evaluator(src)
        assert res == "{}"

    def test_bound_function_application(self, evaluator):
        src = "let foo = fun x -> x; (foo {})"
        res = evaluator(src)
        assert res == "{}"

    def test_let_over_lambda(self, evaluator):
        src = "(let x = {} in fun y -> x) true"
        res = evaluator(src)
        assert res == "{}"

    def test_conditional(self, evaluator):
        src = "if true then false else true"
        res = evaluator(src)
        assert res == "false"

    def test_conditional_with_different_types(self, evaluator):
        src = "if false then true else {b=true;c=true}"
        res = evaluator(src)
        assert res == "{b=true; c=true}"

    def test_toplevel_recursive_binding(self, evaluator):
        src = """
            let rec turn = fun x -> foo true
                and foo = fun x -> if x then x else turn x;
            foo false
        """
        res = evaluator(src)
        assert res == "true"

    def test_record_and_field_access(self, evaluator):
        src = """
            let data = {alpha=true;beta=false;gamma={}};
            data.beta
        """
        res = evaluator(src)
        assert res == "false"

    def test_case_and_match(self, evaluator):
        src = "match `Bar true with | `Foo x -> false | `Bar y -> y"
        res = evaluator(src)
        assert res == "true"

    def test_let_expression(self, evaluator):
        src = "(let x = {} in x)"
        res = evaluator(src)
        assert res == "{}"

    def test_letrec_expression(self, evaluator):
        src = "(let r = true in (let rec foo = fun x -> if x then foo false else r in foo true))"
        res = evaluator(src)
        assert res == "true"

    def test_let_polymorphism(self, evaluator):
        src = """
            let id = fun x -> x;
            
            let a = (id {a={}}).a;
            let b = (id {b=true}).b;
            
            {a=a; b=b}
        """
        res = evaluator(src)
        assert res == "{a={}; b=true}"

    def test_letrec_polymorphism(self, evaluator):
        src = """
            let rec id = fun x -> if true then x else (id x);
            
            let a = (id {a={}}).a;
            let b = (id {b=true}).b;
            
            {a=a; b=b}
        """
        res = evaluator(src)
        assert res == "{a={}; b=true}"

    def test_fibonacci(self, evaluator):
        src = """
            let rec fib = fun n -> 
                if n < 2
                then 1
                else fib (n - 1) + fib (n - 2);
            fib(10)
        """
        res = evaluator(src)
        assert res == "89"

    def test_procedures(self, evaluator):
        src = "(proc x -> do x + 10; x + 2 end) 5"
        res = evaluator(src)
        assert res == "7"

    def test_references(self, evaluator):
        src = "let x = {y=ref 3}; x.y := 5; !x.y + !x.y"
        res = evaluator(src)
        assert res == "10"

    def test_reference_not_allowed(self, evaluator):
        src = "let id = fun x -> x; let x = ref 0; id(x:=1)"
        with pytest.raises(TypeError, match="Unusable"):
            evaluator(src)

        src = "let x = ref 0; let y = x := 1; 2"
        with pytest.raises(TypeError, match="Unusable"):
            evaluator(src)

    def test_procedure_in_functional_context(self, evaluator):
        src = "let foo = proc x -> x; let bar = fun x -> (foo x); bar"
        with pytest.raises(TypeError, match="TODO"):
            res = evaluator(src)

    def test_mutation_in_functional_context(self, evaluator):
        src = "let foo = fun x -> x := 0; foo"
        with pytest.raises(TypeError, match="TODO"):
            res = evaluator(src)


def transform_python_result(res) -> str:
    match res:
        case None:
            return str(res)
        case bool():
            return str(res).lower()
        case int():
            return str(res)
        case dict():
            return (
                "{"
                + "; ".join(f"{k}={transform_python_result(v)}" for k, v in res.items())
                + "}"
            )
        case interpreter.Function():
            return f"<fun>"
        case interpreter.Cell(val):
            return f"(ref {transform_python_result(val)})"
        case _:
            raise NotImplementedError(res)
