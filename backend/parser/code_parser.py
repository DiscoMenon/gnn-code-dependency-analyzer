import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from pathlib import Path

PY_LANGUAGE = Language(tspython.language())

class PythonFileParser:
    def __init__(self):
        self.parser = Parser(PY_LANGUAGE)

    def parse_file(self, filepath: Path) -> dict:
        with open(filepath, "rb") as f:
            source = f.read()

        tree = self.parser.parse(source)
        root = tree.root_node

        return {
            "imports": self._extract_imports(root, source),
            "functions": self._extract_functions(root),
            "classes": self._extract_classes(root),
        }

    def _extract_imports(self, root, source: bytes) -> list[str]:
        imports = []
        for node in self._walk(root):
            # handles: import os, import os.path
            if node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        imports.append(child.text.decode())
                    elif child.type == "aliased_import":
                        for n in child.children:
                            if n.type == "dotted_name":
                                imports.append(n.text.decode())
                                break

            # handles: from os import path, from . import utils
            elif node.type == "import_from_statement":
                module = None
                for child in node.children:
                    if child.type == "dotted_name":
                        module = child.text.decode()
                        break
                    elif child.type == "relative_import":
                        module = child.text.decode()
                        break
                if module:
                    imports.append(module)

        return imports

    def _extract_functions(self, root) -> list[str]:
        return [
            node.child_by_field_name("name").text.decode()
            for node in self._walk(root)
            if node.type == "function_definition"
            and node.child_by_field_name("name")
        ]

    def _extract_classes(self, root) -> list[str]:
        return [
            node.child_by_field_name("name").text.decode()
            for node in self._walk(root)
            if node.type == "class_definition"
            and node.child_by_field_name("name")
        ]

    def _walk(self, node):
        yield node
        for child in node.children:
            yield from self._walk(child)