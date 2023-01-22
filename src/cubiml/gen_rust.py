from biunification.type_checker import TypeCheckerCore
from cubiml import abstract_syntax as ast, type_heads


def compile_script(script: ast.Script, type_mapping, engine: TypeCheckerCore) -> str:
    result = []
    for stmt in script.statements:
        result.append(compile_toplevel(stmt, type_mapping, engine))
    return "\n\n".join(result)


def compile_toplevel(
    stmt: ast.ToplevelItem, type_mapping, engine: TypeCheckerCore
) -> str:
    match stmt:
        case ast.Expression() as expr:
            return compile_expr(expr, type_mapping, engine)
        case ast.DefineLet(var, val):
            cval = compile_expr(val, type_mapping, engine)
            ty = compile_type(val, type_mapping, engine)
            return f"let {var}: {ty} = {cval};"
        case _:
            raise NotImplementedError(stmt)


def compile_expr(expr: ast.Expression, type_mapping, engine: TypeCheckerCore) -> str:
    match expr:
        case ast.Literal(True):
            return "true"
        case ast.Literal(False):
            return "false"
        case _:
            raise NotImplementedError(expr)


def compile_type(expr: ast.Expression, type_mapping, engine: TypeCheckerCore) -> str:
    t = type_mapping[id(expr)]
    match engine.types[t]:
        case type_heads.VBool():
            return "bool"
        case other:
            raise NotImplementedError(other)
