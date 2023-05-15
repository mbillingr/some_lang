import contextlib

from eopl_explicit_refs import abstract_syntax as ast, type_impls
from eopl_explicit_refs.vtable_manager import VtableManager


def transform_virtuals(node: ast.Program):
    return node.transform(Visitor().visit)


class Visitor:
    def __init__(self):
        self.vtm = VtableManager()
        self.interfaces = {}
        self.type_ifs = {}
        self.static_functions = []
        self.static_funcnames = {}
        self.vtables = {}

    def visit(self, node: ast.AstNode):
        match node:
            case ast.Program():
                raise TypeError("Can't virtualize raw program. Run through type checker first")

            case ast.CheckedProgram(mods, exp):
                self.__init__()
                mod_out = ast.transform_dict_values(mods, self.visit)
                exp_out = exp.transform(self.visit)
                return ast.ExecutableProgram(mod_out, exp_out, self.vtables)

            case ast.CheckedModule(name, submodules,  imports, types, impls):
                for name, ty in types.items():
                    if not isinstance(ty, type_impls.InterfaceType):
                        continue
                    self.interfaces[ty.fully_qualified_name] = self.vtm.assign_virtuals(ty.methods.keys())
                return node.default_transform(self.visit)

            case ast.Interface(name, methods):
                raise NotImplementedError("Unexpected interface declaration")

            case ast.ImplBlock(None):
                node_out = node.default_transform(self.visit)
                self._register_functions(node_out)
                return node_out

            case ast.ImplBlock(interface, type_name, methods):
                self.type_ifs.setdefault(type_name, set()).add(interface)
                node_out = node.default_transform(self.visit)
                self._register_functions(node_out)

                for qual_ifname in self.type_ifs[type_name]:
                    for method_name, (tbl, idx) in self.interfaces[qual_ifname].items():
                        self.vtables.setdefault(type_name, {}).setdefault(tbl, {})[idx] = self.static_funcnames[
                            method_name
                        ]

                return node_out

            case ast.GetNamedVirtual(obj, interface, method_name):
                tbl, idx = self.interfaces[interface][method_name]
                return ast.GetVirtual(obj.transform(self.visit), tbl, idx)

            case _:
                return NotImplemented

    def _register_functions(self, node_out):
        for method_name, method_body in node_out.methods.items():
            self.static_funcnames[method_name] = len(self.static_functions)
            self.static_functions.append(method_body)
