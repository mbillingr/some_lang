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
                _ = ast.transform_dict_values(mods, self.visit)
                exp_out = exp.transform(self.visit)
                return ast.ExecutableProgram(self.static_functions, exp_out, self.vtables)

            case ast.CheckedModule(name, types, impls):
                for name, ty in types.items():
                    if not isinstance(ty, type_impls.InterfaceType):
                        continue
                    self._register_interface(ty)
                return node.default_transform(self.visit)

            case ast.Interface(name, methods):
                raise NotImplementedError("Unexpected interface declaration")

            case ast.ImplBlock(None):
                node_out = node.default_transform(self.visit)
                self._register_functions(node_out, node.type_name)
                return node_out

            case ast.ImplBlock(interface, type_name, methods):
                self.type_ifs.setdefault(type_name, set()).add(interface)
                node_out = node.default_transform(self.visit)
                self._register_functions(node_out, node.type_name, interface)

                for qual_ifname in self.type_ifs[type_name]:
                    for method_name, (tbl, idx) in self.interfaces[qual_ifname].items():
                        self.vtables.setdefault(type_name, {}).setdefault(tbl, {})[idx] = self.static_funcnames[
                            self.fully_qualified_method_name(method_name, node.type_name, qual_ifname)
                        ]

                return node_out

            case ast.GetMethod(name):
                return ast.GetMethod(self.static_funcnames[name])

            case ast.GetNamedVirtual(obj, interface, method_name):
                tbl, idx = self.interfaces[interface][method_name]
                return ast.GetVirtual(obj.transform(self.visit), tbl, idx)

            case _:
                return NotImplemented

    def _register_interface(self, intf):
        if intf.fully_qualified_name in self.interfaces:
            return
        self.interfaces[intf.fully_qualified_name] = self.vtm.assign_virtuals(intf.methods.keys())

    def _register_functions(self, node_out, type_name: str, interface_name: str = ""):
        for method_name, method_body in node_out.methods.items():
            name = self.fully_qualified_method_name(method_name, type_name, interface_name)
            if name not in self.static_funcnames:
                self.static_funcnames[name] = len(self.static_functions)
                self.static_functions.append(method_body)

    def fully_qualified_method_name(self, method_name: str, type_name: str, interface_name: str = "") -> str:
        if interface_name:
            return f"{type_name}.{interface_name}.{method_name}"
        else:
            return f"{type_name}..{method_name}"
