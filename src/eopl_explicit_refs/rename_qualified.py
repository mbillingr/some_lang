import contextlib

from eopl_explicit_refs import abstract_syntax as ast
from eopl_explicit_refs.vtable_manager import VtableManager


def rename_qualified(node: ast.Program):
    return node.transform(Visitor().visit)


class Visitor:
    def __init__(self):
        self.path = []
        self.decl_env = {}

    def visit(self, node: ast.AstNode):
        match node:
            case ast.Program(mod, exp):
                self.__init__()
                self.path.append(mod.name)
                mod_out = mod.default_transform(self.visit)
                exp_out = exp.transform(self.visit)
                return ast.Program(mod_out, exp_out)

            case ast.Module():
                old_decl_env = self.decl_env
                self.decl_env = {}
                self.path.append(node.name)
                node_out = node.default_transform(self.visit)
                node_out.name = self.make_qualname()
                self.path.pop()
                self.decl_env = old_decl_env
                return node_out

            case ast.Import():
                for imported_qualname_parts in node.iter():
                    name = imported_qualname_parts[-1]
                    self.decl_env[name] = self.make_qualname(*imported_qualname_parts)
                return node.default_transform(self.visit)

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

            case ast.ImplBlock(interface, type_name, methods):
                qual_ifname = interface and self.decl_env[interface]
                qual_tyname = self.decl_env[type_name]
                node_out = node.default_transform(self.visit)
                node_out.interface = qual_ifname
                node_out.type_name = qual_tyname
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

            case _:
                return NotImplemented

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
