import cubiml.tokenizer
from cubiml import type_checker, interpreter, parser2, tokenizer

tckr = type_checker.TypeChecker()
intr = interpreter.Interpreter()


def read_more(src):
    last_line = src
    while True:
        try:
            token_stream = tokenizer.default_tokenizer(src)
            return parser2.parse_toplevel(token_stream)
        except cubiml.tokenizer.UnexpectedEnd:
            if src and not last_line:
                raise cubiml.tokenizer.UnexpectedEnd(tokenizer.Span.make_eof(src))
        except cubiml.tokenizer.UnexpectedToken as e:
            tok, kind, span = e.args[0]
            # unexpected DEDENTs at the end are likely due to incomplete inputs
            if kind != tokenizer.TokenKind.END_BLOCK and span.start < len(src):
                raise

        while True:
            last_line = input("| ")
            src += "\n" + last_line

            if not last_line:
                break

            # is the last line indented?
            if last_line.lstrip() == last_line:
                break


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
    except tokenizer.UnexpectedEnd as e:
        span = e.args[0]
        print(f"{type(e).__name__}")
        print(span.show_line())
    except tokenizer.UnexpectedToken as e:
        print(e)
    # except Exception as e:
    #    print(f"{type(e).__name__}: {e}")
