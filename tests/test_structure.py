"""The project layout: backend is the logic, frontend is everything you see."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_frontend_imports_nothing_from_backend():
    for path in (ROOT / "frontend").rglob("*.py"):
        assert not any(n == "backend" or n.startswith("backend.") for n in imports(path)), path.name


def test_no_ui_code_left_in_backend():
    assert not (ROOT / "backend" / "ui").exists()
    assert (ROOT / "frontend" / "screen" / "index.html").is_file()
