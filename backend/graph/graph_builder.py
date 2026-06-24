import networkx as nx
from pathlib import Path
from backend.parser.code_parser import PythonFileParser

class CodeGraphBuilder:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.parser = PythonFileParser()
        self.G = nx.DiGraph()

    def build(self) -> nx.DiGraph:
        py_files = list(self.repo_path.rglob("*.py"))
        print(f"Found {len(py_files)} Python files")

        # First pass: add all file nodes
        for filepath in py_files:
            rel_path = str(filepath.relative_to(self.repo_path))
            self.G.add_node(rel_path, type="file", path=rel_path)

        # Second pass: parse and add edges
        for filepath in py_files:
            rel_path = str(filepath.relative_to(self.repo_path))
            try:
                parsed = self.parser.parse_file(filepath)
            except Exception as e:
                print(f"  Skipped {rel_path}: {e}")
                continue

            # Add function/class nodes
            for fn in parsed["functions"]:
                node_id = f"{rel_path}::{fn}"
                self.G.add_node(node_id, type="function", parent_file=rel_path)
                self.G.add_edge(rel_path, node_id, type="contains")

            for cls in parsed["classes"]:
                node_id = f"{rel_path}::{cls}"
                self.G.add_node(node_id, type="class", parent_file=rel_path)
                self.G.add_edge(rel_path, node_id, type="contains")

            # Add import edges (file → file, if target exists in repo)
            for imp in parsed["imports"]:
                target = self._resolve_import(imp)
                if target and self.G.has_node(target):
                    self.G.add_edge(rel_path, target, type="imports")

        return self.G

    def _resolve_import(self, import_str: str) -> str | None:
        """Match import strings directly against existing NetworkX file nodes"""
        # Convert 'requests.utils' -> ['requests', 'utils']
        parts = import_str.lstrip(".").split(".")
        if not parts or parts == [""]:
            return None

        # Create possible matching suffixes like 'requests/utils.py' or 'utils.py'
        suffix_file = "/".join(parts) + ".py"
        suffix_init = "/".join(parts) + "/__init__.py"

        # Look through the file nodes we ALREADY found in the repo
        for node in self.G.nodes:
            if self.G.nodes[node].get("type") == "file":
                # Normalize paths to use forward slashes for cross-platform safety
                normalized_node = node.replace("\\", "/")
                
                if normalized_node.endswith(suffix_file) or normalized_node.endswith(suffix_init):
                    return node  # Return the exact node key used in the graph
                    
        return None