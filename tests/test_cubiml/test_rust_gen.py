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
    # ast = parser.parse_script("{a=true;f=false}.a")
    # ast = parser.parse_script("""
    #    let take_a = fun r -> r.a;
    #    take_a {a=true; b=false}
    # """)
    # ast = parser.parse_script("if true then {a=false} else {a=true}")
    ast = parser.parse_script(
        """
           ((fun x -> if x.a then x.b else x.c) {a=true;b={z=true};c={z=false}}).z
       """
    )
    tck = type_checker.TypeChecker()
    typemap = tck.check_script(ast)
    print(tck.engine)
    runner = gen_rust.Runner(typemap, tck.engine)
    assert runner.run_script(ast) == "bla"


def test_toplevel_binding():
    src = "let x = false; x"
    res = eval_in_rust(src)
    assert res == "false"


def test_function_application():
    src = "(fun x -> x) {}"
    res = eval_in_rust(src)
    assert res == "{}"


def test_conditional():
    src = "if true then false else true"
    res = eval_in_rust(src)
    assert res == "false"


def test_conditional_with_different_types():
    src = "if false then {a=true;b=true} else {b=true;c=true}"
    res = eval_in_rust(src)
    assert res == "{c=true; b=true}"


def test_toplevel_recursive_binding():
    src = """
        let rec turn = fun x -> foo true
            and foo = fun x -> if x then x else turn x;
        turn
    """
    res = eval_in_rust(src)
    assert res == "true"


def test_record_and_field_access():
    src = """
        let data = {alpha=true;beta=false;gamma={}};
        data.beta
    """
    res = eval_in_rust(src)
    assert res == "false"


def test_case_and_match():
    raise NotImplementedError()


def test_let_expression():
    src = "(let x = {} in x)"
    res = eval_in_rust(src)
    assert res == "{}"


def test_letrec_expression():
    src = "(let rec foo = fun x -> if x then foo false else true in foo true)"
    res = eval_in_rust(src)
    assert res == "true"


def eval_in_rust(src: str) -> str:
    ast = parser.parse_script(src)
    tck = type_checker.TypeChecker()
    typemap = tck.check_script(ast)
    runner = gen_rust.Runner(typemap, tck.engine)
    return runner.run_script(ast)
