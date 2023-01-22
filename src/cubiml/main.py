from cubiml import type_checker, interpreter, parser

tckr = type_checker.TypeChecker()
intr = interpreter.Interpreter()


def read_more(src):
    while True:
        try:
            return parser.parse_script(src)
        except parser.ParseException as e:
            if e.loc != len(src):
                raise
        src += "\n" + input("| ")


while True:
    try:
        src = input("> ")
        ast = read_more(src)
        ty = tckr.check_script(ast)
        print(tckr.engine)
        print(tckr.bindings.m)
        if ty is not None:
            print(f"t{ty}")
        val = intr.run_script(ast)
        if val is not None:
            print(val)
    except EOFError:
        break
    except Exception as e:
        print(e)
