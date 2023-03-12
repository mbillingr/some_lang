from tinyml import abstract_syntax as ast
from tinyml.type_checker import infer, empty_tenv, check, FunctionType


def test_infer_literals():
    assert infer(ast.Literal(True), empty_tenv()) == bool
    assert infer(ast.Literal(42), empty_tenv()) == int


def test_infer_variables():
    assert infer(ast.Reference("x"), empty_tenv().insert("x", bool)) == bool


def test_check_function():
    check(FunctionType(int, int), ast.Function("x", ast.Reference("x")), empty_tenv())

    check(
        FunctionType(int, FunctionType(int, bool)),
        ast.Function("x", ast.Function("y", ast.Literal(False))),
        empty_tenv(),
    )


def test_infer_application():
    assert (
        infer(
            ast.Application(ast.Reference("f"), ast.Literal(0)),
            empty_tenv().insert("f", FunctionType(int, bool)),
        )
        == bool
    )

    assert (
        infer(
            ast.Application(ast.Function("x", ast.Reference("x")), ast.Literal(0)),
            empty_tenv(),
        )
        == int
    )
