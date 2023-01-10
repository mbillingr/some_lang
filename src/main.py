from some_lang import parser
from some_lang.interpreter import run_module
from some_lang.type_checker import check_module

if __name__ == "__main__":
    ast = parser.parse_module(
        """
def not(Int) -> Int:
    not(0) = 1
    not(x) = 0
    
def foo(Int) -> Int:
    foo(x) = x
    
def bar(Int) -> Int:
    bar(x) = x
    
def baz(Int) -> Int:
    baz(x) = (bar x)
    
print (not (not 3))
"""
    )
    print(ast)
    check_module(ast)
    run_module(ast)

    # current idea:
    #   - use type system to distinguish between pure and imperative functions
    #     -> two different function types
    #     -> application in pure context expects pure function
    #     -> application in imperative context expects either function
    #     -> can we have (do we need) statements?
