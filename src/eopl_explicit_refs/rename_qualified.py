import functools

from eopl_explicit_refs import abstract_syntax as ast


def rename_qualified(node: ast.Program):
    return node.transform(Visitor().visit)


class Visitor:
    def __init__(self):
        self.path = []
        self.decl_env = {}
        self.local_env = set()

    def visit(self, node: ast.AstNode):
        match node:
            case ast.Program(mod, exp):
                self.__init__()
                self.path.append(mod.name)
                mod_out = self.transform_module(mod)
                exp_out = exp.transform(self.visit)
                return ast.Program(mod_out, exp_out)

            case ast.Module():
                old_decl_env = self.decl_env
                old_path = self.path.copy()
                self.decl_env = {}
                self.path.extend(node.name.split("."))
                node_out = self.transform_module(node)
                self.path = old_path
                self.decl_env = old_decl_env
                return node_out

            case ast.AbsoluteImport(module, name):
                self.decl_env[name] = module + "." + name
                return node

            case ast.Import():
                raise NotImplementedError(
                    "All imports should be absolute at this point"
                )

            case ast.Interface(name, methods):
                qual_ifname = self.add_decl(name)
                node_out = node.default_transform(self.visit)
                node_out.name = qual_ifname
                return node_out

            case ast.RecordDecl(name):
                qual_tyname = self.add_decl(name)
                node_out = node.default_transform(self.visit)
                node_out.name = qual_tyname
                return node_out

            case ast.ImplBlock(interface, type_name):
                qual_ifname = interface and self.decl_env[interface]
                qual_tyname = self.decl_env[type_name]
                node_out = node.default_transform(self.visit)
                node_out.interface = qual_ifname
                node_out.type_name = qual_tyname
                return node_out

            case ast.FunctionDefinition(name):
                qual_fname = self.decl_env[name]
                node_out = node.default_transform(self.visit)
                node_out.name = qual_fname
                return node_out

            case ast.Generic(tvars, item):
                tvars_out = [
                    (tv, tuple(self.decl_env[c] for c in cs)) for tv, cs in tvars
                ]
                item_out = item.transform(self.visit)
                return ast.Generic(tvars_out, item_out)

            case ast.TypeRef(typename):
                node_out = node.default_transform(self.visit)
                try:
                    qual_tyname = self.decl_env[typename]
                    node_out.name = qual_tyname
                except KeyError:
                    pass
                return node_out

            case ast.WithInterfaces(obj, typename):
                qual_tyname = self.decl_env[typename]
                node_out = node.default_transform(self.visit)
                node_out.typename = qual_tyname
                return node_out

            case ast.GetNamedVirtual(obj, interface, method_name):
                qual_ifname = self.decl_env[interface]
                node_out = node.default_transform(self.visit)
                node_out.interface = qual_ifname
                return node_out

            case ast.Let(var):
                defined_before = var in self.local_env
                self.local_env.add(var)
                node_out = node.default_transform(self.visit)
                if not defined_before:
                    self.local_env.remove(var)
                return node_out

            case ast.MatchArm(pats, body):
                bound_vars = functools.reduce(
                    lambda a, b: a | b, map(vars_in_pattern, pats)
                )
                undefined_before = bound_vars - self.local_env
                node_out = node.default_transform(self.visit)
                self.local_env -= undefined_before
                return node_out

            case ast.Identifier(name):
                if name not in self.local_env and name in self.decl_env:
                    return ast.ToplevelRef(self.decl_env[name])
                else:
                    return node

            case _:
                return NotImplemented

    def transform_module(self, node: ast.Module) -> ast.Module:
        ast.transform_collection(node.funcs, self.declare_function),

        node_out = ast.Module(
            self.make_qualname(),
            ast.transform_dict_values(node.submodules, self.visit),
            ast.transform_collection(self.flatten_imports(node.imports), self.visit),
            ast.transform_collection(node.interfaces, self.visit),
            ast.transform_collection(node.records, self.visit),
            ast.transform_collection(node.impls, self.visit),
            ast.transform_collection(node.funcs, self.visit),
        )
        return node_out

    def declare_function(self, node: ast.AstNode):
        match node:
            case ast.FunctionDefinition(name):
                self.add_decl(name)
                return node
            case _:
                return node.default_transform(self.declare_function)

    def flatten_imports(self, imports: list[ast.Import]) -> list[ast.Import]:
        imports_out = []
        for imp in imports:
            match imp:
                case ast.AbsoluteImport():
                    imports_out.extend(imp)
                case ast.RelativeImport(_, _, offset):
                    offset = len(self.path) + offset
                    for [*path, thing] in imp.iter():
                        imports_out.append(
                            ast.AbsoluteImport(
                                ".".join(self.path[:offset] + path), thing
                            )
                        )
                case ast.NestedImport():
                    for [*path, thing] in imp.iter():
                        imports_out.append(ast.AbsoluteImport(".".join(path), thing))
        return imports_out

    def _register_functions(self, node_out):
        for method_name, method_body in node_out.methods.items():
            method_qualname = self.make_qualname(node_out.type_name, method_name)
            self.static_funcnames[method_qualname] = len(self.static_functions)
            self.static_functions.append(method_body)

    def make_qualname(self, *args: str) -> str:
        return ".".join((*self.path, *args))

    def add_decl(self, name: str) -> str:
        qualname = self.make_qualname(name)
        self.decl_env[name] = qualname
        return qualname


def vars_in_pattern(pat: ast.Pattern) -> set[str]:
    match pat:
        case ast.BindingPattern(name):
            return {name}
        case ast.LiteralPattern(val):
            return set()
        case ast.ListConsPattern(car, cdr):
            return vars_in_pattern(car) | vars_in_pattern(cdr)
        case ast.EmptyListPattern():
            return set()
        case other:
            raise NotImplementedError(other)
