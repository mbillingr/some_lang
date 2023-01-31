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


@pytest.mark.parametrize("evaluator", [eval_in_python, eval_in_rust, eval_in_rust2])
class TestLanguage:
    def test_just_a_literal(self, evaluator):
        src = "true"
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

    def test_conditional(self, evaluator):
        src = "if true then false else true"
        res = evaluator(src)
        assert res == "false"

    def test_conditional_with_different_types(self, evaluator):
        src = "if false then {a=true;b=true} else {b=true;c=true}"
        res = evaluator(src)
        assert res == "{b=true; c=true}"

    def test_toplevel_recursive_binding(self, evaluator):
        src = """
            let rec turn = fun x -> foo true
                and foo = fun x -> if x then x else turn x;
            turn
        """
        res = evaluator(src)
        assert res == "<fun>"

    def test_record_and_field_access(self, evaluator):
        src = """
            let data = {alpha=true;beta=false;gamma={}};
            data.beta
        """
        res = evaluator(src)
        assert res == "false"

    def test_case_and_match(self, evaluator):
        src = "match `Bar {} with | `Foo x -> false | `Bar y -> true"
        res = evaluator(src)
        assert res == "true"

    def test_let_expression(self, evaluator):
        src = "(let x = {} in x)"
        res = evaluator(src)
        assert res == "{}"

    def test_letrec_expression(self, evaluator):
        src = "(let rec foo = fun x -> if x then foo false else true in foo true)"
        res = evaluator(src)
        assert res == "true"

    def test_let_polymorphism(self, evaluator):
        src = """
            let id = fun x -> x;
            
            let a = (id {a={}}).a;
            let b = (id {b={}}).b;
            
            {a={}; b={}}
        """
        res = evaluator(src)
        assert res == "{a={}; b={}}"


def transform_python_result(res) -> str:
    match res:
        case bool():
            return str(res).lower()
        case dict():
            return (
                "{"
                + "; ".join(f"{k}={transform_python_result(v)}" for k, v in res.items())
                + "}"
            )
        case interpreter.Function():
            return f"<fun>"
        case _:
            raise NotImplementedError(res)
