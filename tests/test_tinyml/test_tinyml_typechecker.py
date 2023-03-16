from tinyml import abstract_syntax as ast
from tinyml.type_checker import infer, empty_tenv, check, FunctionType, Int, Bool


def test_infer_literals():
    assert infer(ast.Literal(True), empty_tenv()) == Bool()
    assert infer(ast.Literal(42), empty_tenv()) == Int()


def test_infer_variables():
    assert infer(ast.Reference("x"), empty_tenv().extend("x", bool)) == bool


def test_check_function():
    check(FunctionType(Int(), Int()), ast.Function("x", ast.Reference("x")), empty_tenv())

    check(
        FunctionType(Int(), FunctionType(Int(), Bool())),
        ast.Function("x", ast.Function("y", ast.Literal(False))),
        empty_tenv(),
    )


def test_infer_application():
    assert (
        infer(
            ast.Application(ast.Reference("f"), ast.Literal(0)),
            empty_tenv().extend("f", FunctionType(Int(), Bool())),
        )
        == Bool()
    )

    assert (
        infer(
            ast.Application(ast.Function("x", ast.Reference("x")), ast.Literal(0)),
            empty_tenv(),
        )
        == Int()
    )
