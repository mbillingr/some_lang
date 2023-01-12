from typing import TypeAlias

from some_lang import ast, type_heads
from some_lang.biunification.type_checker import Value, TypeCheckerCore, Use
from some_lang.env import EmptyEnv, Env


def eval_vtype(texp: ast.TypeExpression, engine: TypeCheckerCore) -> Value:
    match texp:
        case ast.IntegerType():
            return engine.new_val(type_heads.VInt())
        case _:
            raise NotImplementedError(texp)


def eval_utype(texp: ast.TypeExpression, engine: TypeCheckerCore) -> Use:
    match texp:
        case ast.IntegerType():
            return engine.new_use(type_heads.UInt())
        case _:
            raise NotImplementedError(texp)


def check_module(mod: ast.Module, engine: TypeCheckerCore):
    env: Env[Value] = EmptyEnv()

    for defn in mod.defs:
        # arg = eval_utype(defn.arg, engine)
        # res = (eval_vtype(defn.res, engine),)
        arg_type, arg_bound = engine.var()
        res_type, res_usage = engine.var()
        func_type = engine.new_val(type_heads.VFunc(arg_bound, res_type))

        local_env = env
        for pattern in defn.patterns:
            assert pattern.name == defn.name
            match pattern.pat:
                case ast.IntegerPattern():
                    engine.flow(engine.new_val(type_heads.VInt()), arg_bound)
                case ast.BindingPattern(var):
                    local_env = local_env.extend(var, arg_type)
                    body_type = check_expr(pattern.exp, local_env, engine)
                    engine.flow(body_type, res_usage)

        env = env.extend(defn.name, func_type)

    for stmt in mod.code:
        check_stmt(stmt, env, engine)


def check_expr(expr: ast.Expression, env: Env[Value], engine: TypeCheckerCore) -> Value:
    match expr:
        case ast.Integer():
            return engine.new_val(type_heads.VInt())
        case ast.Reference(var):
            t = env.apply(var)
            if t is None:
                raise NameError("Unbound variable", var)
            return t
        case ast.Application(rator, rand):
            func_type = check_expr(rator, env, engine)
            arg_type = check_expr(rand, env, engine)
            ret_type, ret_bound = engine.var()
            bound = engine.new_use(type_heads.UFunc(arg_type, ret_bound))
            engine.flow(func_type, bound)
            return ret_type
        case _:
            raise NotImplementedError(expr)


def check_stmt(stmt: ast.Statement, env: Env[Value], engine: TypeCheckerCore):
    match stmt:
        case ast.PrintStatement(exp):
            # print accepts any type; just make sure exp does not contain type errors
            t = check_expr(exp, env, engine)
            print("Type:", engine.resolve(t))
        case _:
            raise NotImplementedError(stmt)
