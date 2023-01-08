from some_lang import env, parser
from some_lang.interpreter import evaluate

if __name__ == "__main__":
    ast = parser.parse_program("((lambda (foo)\n  42) 123)")
    print(ast)
    res = evaluate(ast, env.EmptyEnv())
    print(res)
