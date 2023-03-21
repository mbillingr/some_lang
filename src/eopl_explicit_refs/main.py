from eopl_explicit_refs import interpreter, tokenizer, parser
from eopl_explicit_refs.store import is_reference, deref


def print_result(val):
    if is_reference(val):
        print("*", end="")
        print_result(deref(val))
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
result = interpreter.value_of_program(program)

print_result(result)
