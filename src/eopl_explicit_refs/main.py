import sys
from pathlib import Path

from eopl_explicit_refs import interpreter, tokenizer, parser, rename_qualified, type_checker, transform_virtuals
from eopl_explicit_refs.store import PythonStore
from eopl_explicit_refs.type_checker import Context

DEFAULT_SEARCH_PATH = Path(__file__).parent / "sources"
LIB_EXTENSION = ".src"


def main(filename):
    run_file(filename)


def run_file(filename):
    with open(filename) as fd:
        src = fd.read()

    token_stream = tokenizer.default_tokenizer(src)
    program = parser.parse_program(token_stream)
    program = rename_qualified.rename_qualified(program)
    checked = type_checker.check_program(program, context_args={"import_hooks": [lib_loader]})
    execable = transform_virtuals.transform_virtuals(checked)
    runner = interpreter.analyze_program(execable)
    return runner(PythonStore())


def lib_loader(ctx, name):
    try:
        with open(DEFAULT_SEARCH_PATH / ("/".join(name.split(".")) + LIB_EXTENSION)) as fd:
            src = fd.read()
    except FileNotFoundError:
        raise

    token_stream = tokenizer.default_tokenizer(src)
    module = parser.parse_module_body(token_stream, name)
    module = rename_qualified.rename_qualified(module)
    checked, _ = type_checker.check_module(module, ctx)
    return checked



if __name__ == "__main__":
    main(*sys.argv[1:])
