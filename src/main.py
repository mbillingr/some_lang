from some_lang import parser
from some_lang.biunification.type_checker import TypeCheckerCore
from some_lang.interpreter import run_module
from some_lang.lang_frontend import Context
from some_lang.type_checker import check_module

if __name__ == "__main__":
    src = """
def ident(?) -> ?:
    ident(x) = x
    
print (0? (ident 7))
"""

    ctx = Context()
    ctx.init_default_env()
    ctx = ctx.module(src)
    print(ctx.engine)
    print("----------")

    ctx = Context()
    ctx.init_default_env()
    print(ctx.compile_expr("((lambda (x) x) (lambda (a) (lambda (b) a)))"))
    raise NotImplementedError(
        "correctly annotate type variables... arg: (_0 -> (_1 -> _0))"
    )

    # current idea:
    #   - use type system to distinguish between pure and imperative functions
    #     -> two different function types
    #     -> application in pure context expects pure function
    #     -> application in imperative context expects either function
    #     -> can we have (do we need) statements?
