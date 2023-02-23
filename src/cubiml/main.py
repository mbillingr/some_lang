from cubiml import type_checker, interpreter, parser2, tokenizer

tckr = type_checker.TypeChecker()
intr = interpreter.Interpreter()


def read_more(src):
    while True:
        try:
            token_stream = tokenizer.default_tokenizer(src)
            return parser2.parse_toplevel(token_stream)
        except parser2.UnexpectedEnd as e:
            pass
        src += "\n" + input("| ")


while True:
    try:
        src = input("> ")
        ast = read_more(src)
        ty = tckr.check_script(ast)
        print(tckr.ctx.engine)
        print(tckr.ctx.bindings.m)
        if ty is not None:
            print(f"t{ty}")
        val = intr.run_script(ast)
        if val is not None:
            print(val)
    except EOFError:
        break
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
