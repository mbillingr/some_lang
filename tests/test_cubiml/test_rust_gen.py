from cubiml import parser, type_checker, gen_rust


def test_stuff():
    ast = parser.parse_script("let not = (fun x -> if x then false else true) true")
    tck = type_checker.TypeChecker()
    typemap = tck.check_script(ast)
    print(tck.engine)
    assert gen_rust.compile_script(ast, typemap, tck.engine) == ""
