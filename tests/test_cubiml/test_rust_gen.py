from cubiml import parser, type_checker, gen_rust


def test_stuff():
    #ast = parser.parse_script("let z = (fun x -> x) true")
    ast = parser.parse_script("let not = (fun x -> if x then false else true) true")
    tck = type_checker.TypeChecker()
    typemap = tck.check_script(ast)
    print(tck.engine)
    compiler = gen_rust.Compiler()
    compiler.compile_script(ast, typemap, tck.engine)
    print(compiler.finalize())
    assert compiler.finalize() == ""
