from tinyml import type_checker, tokenizer, parser


def read_more(src):
    last_line = src
    while True:
        if not src.endswith(":"):
            try:
                token_stream = tokenizer.default_tokenizer(src, implicit_block=True)
                return parser.parse_expr(token_stream)
            except tokenizer.UnexpectedEnd:
                if src and not last_line:
                    raise tokenizer.UnexpectedEnd(tokenizer.Span.make_eof(src))
            except tokenizer.UnexpectedToken as e:
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


tenv = type_checker.empty_tenv()

while True:
    try:
        src = input("> ")
        ast = read_more(src)
        ty = type_checker.infer(ast, tenv)
        print("Inferred type:", ty)
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
