def gen_sym(name: str = "") -> str:
    global _count
    _count += 1
    return f"{name}#{_count}"


_count = 0
