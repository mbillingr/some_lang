from unification import substitution as s
from unification import unify as u


def test_occurs_self_var():
    v = s.Var()
    assert u.occurs(v, v)


def test_not_occurs_different_var():
    assert not u.occurs(s.Var(), s.Var())


def test_occurs_in_list():
    v = s.Var()
    assert u.occurs(v, [1, 2, v, 4])
    assert not u.occurs(v, [1, 2, 3, 4])


def test_unify_two_vars():
    v1, v2 = s.Var(), s.Var()
    subs = u.unify(v1, v2, s.Substitution())
    assert subs.apply(v2) is v1 or subs.apply(v1) is v2


def test_unify_list():
    v1, v2 = s.Var(), s.Var()
    subs = u.unify([v1, 2], [1, v2], s.Substitution())
    assert subs.apply(v1) == 1
    assert subs.apply(v2) == 2


def test_indirect_unification():
    v1, v2 = s.Var(), s.Var()
    subs = u.unify(v1, int, s.Substitution())
    subs = u.unify(v1, v2, subs)
    assert subs.apply(v1) == int
    assert subs.apply(v2) == int
