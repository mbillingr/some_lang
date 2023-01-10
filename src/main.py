from some_lang import env, parser
from some_lang.interpreter import evaluate, run_module

if __name__ == "__main__":
    ast = parser.parse_module(
        """
def not(Int) -> Int:
    not(0) = 1
    not(x) = 0

print (not (not 3))
"""
    )
    print(ast)
    run_module(ast)

    # current idea:
    #   - use type system to distinguish between pure and imperative functions
    #     -> need working type checking/inference first
    #     -> two different function types
    #     -> application in pure context expects pure function
    #     -> application in imperative context expects either function
    #     -> can we have (do we need) statements?
