from eopl_explicit_refs import interpreter, tokenizer, parser
from eopl_explicit_refs.store import PythonStore


store = PythonStore()


def print_result(val):
    if store.is_reference(val):
        print("*", end="")
        print_result(store.deref(val))
    else:
        print(val)


src_lines = []
while True:
    try:
        src_lines.append(input())
    except EOFError:
        break

token_stream = tokenizer.default_tokenizer("\n".join(src_lines))
program = parser.parse_program(token_stream)
runner = interpreter.analyze_program(program)
result = runner(store)

print_result(result)
