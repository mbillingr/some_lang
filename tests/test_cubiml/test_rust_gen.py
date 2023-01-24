from cubiml import parser, type_checker, gen_rust


def test_stuff():
    # ast = parser.parse_script("let z = (fun x -> x) true")
    # ast = parser.parse_script("let not = (fun x -> if x then false else true) true")
    # ast = parser.parse_script(
    #    """
    #    let twice = fun f -> fun x -> (f (f x));
    #    let id = fun x -> x;
    #    (twice (twice id)) true
    # """
    # )
    #ast = parser.parse_script("{a=true;f=false}.a")
    #ast = parser.parse_script("""
    #    let take_a = fun r -> r.a;
    #    take_a {a=true; b=false}
    #""")
    ast = parser.parse_script("""
            (fun x -> if x.a then x.b else x.c) {a=true;b={};c={}}
        """)
    tck = type_checker.TypeChecker()
    typemap = tck.check_script(ast)
    print(tck.engine)
    compiler = gen_rust.Compiler(typemap, tck.engine)
    compiler.compile_script(ast)
    print(compiler.finalize())
    assert compiler.finalize() == ""
