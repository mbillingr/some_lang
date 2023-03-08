import unification.substitution as s


def test_apply_empty_subst():
    tv = s.Var()
    assert s.Substitution().apply(tv) == tv


def test_apply_empty_subst_to_tvar():
    tv = s.Var()
    subst = s.Substitution().extend(tv, int)
    assert subst.apply(tv) == int


def test_no_occurrence_invariant():
    tv1 = s.Var()
    tv2 = s.Var()
    subst1 = s.Substitution().extend(tv1, tv2)
    subst2 = subst1.extend(tv2, int)
    assert subst2.apply(tv1) != tv2
    assert subst2.apply(tv1) == int


def test_product_type_substitution():
    tv1 = s.Var()
    subst = s.Substitution().extend(tv1, int)
    assert subst.apply([tv1, tv1]) == [int, int]


def test_product_type_nooccurrence():
    tv1 = s.Var()
    tv2 = s.Var()
    subst1 = s.Substitution().extend(tv1, [tv2, tv2])
    subst2 = subst1.extend(tv2, int)
    assert subst2.apply(tv1) == [int, int]
