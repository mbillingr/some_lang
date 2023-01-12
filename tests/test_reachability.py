from some_lang.biunification.reachability import Reachability as R


def test_single_edge():
    r = R()
    a = r.add_node()
    b = r.add_node()

    r.add_edge(a, b)

    assert a in r.upsets[b]
    assert b in r.downsets[a]


def test_reachability():
    r = R()
    a = r.add_node()
    b = r.add_node()
    c = r.add_node()
    d = r.add_node()
    r.add_edge(a, b)
    r.add_edge(c, d)

    r.add_edge(b, c)

    assert c in r.downsets[a]
    assert d in r.downsets[a]
    assert d in r.downsets[b]
    assert a in r.upsets[c]
    assert a in r.upsets[d]
    assert b in r.upsets[d]
