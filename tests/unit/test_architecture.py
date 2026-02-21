"""
Structural test: verify that imports respect the layered architecture.

See docs/harness/ARCHITECTURE_RULES.md for the full dependency specification.
"""

import ast
import pytest
from pathlib import Path

# Map each src/ subdirectory to the set of src/ subdirectories it is allowed to import from.
# Self-imports (e.g. search -> search) are always allowed and handled separately.
ALLOWED_IMPORTS = {
    "ui":         {"mcp_client", "flows", "content", "monitoring", "utils"},
    "flows":      {"search", "graph", "llm", "content", "monitoring", "utils"},
    "search":     {"content", "graph", "utils"},
    "graph":      {"content", "llm", "utils"},
    "llm":        {"utils"},
    "content":    {"utils"},
    "mcp_client": {"flows", "llm", "content", "monitoring", "utils"},
    "monitoring": {"content", "llm", "utils"},
    "utils":      set(),           # Must not import from any src/ module
    "app":        None,            # Not enforced (entry points)
}

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _extract_src_imports(filepath: Path) -> list[tuple[str, int, str]]:
    """
    Parse a Python file and return all imports referencing src.* modules.

    Returns list of (target_module, line_number, import_text) where target_module
    is the first component after 'src.' (e.g. 'flows', 'content').
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if len(parts) >= 2 and parts[0] == "src":
                target = parts[1]
                results.append((target, node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "src":
                    target = parts[1]
                    results.append((target, node.lineno, alias.name))
    return results


def _collect_violations() -> list[str]:
    """Scan all src/ files and collect architecture violations."""
    violations = []

    for subdir in sorted(SRC_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue

        module_name = subdir.name
        allowed = ALLOWED_IMPORTS.get(module_name)

        # Skip modules not in enforcement list
        if allowed is None:
            continue

        for py_file in sorted(subdir.rglob("*.py")):
            rel_path = py_file.relative_to(SRC_DIR.parent)
            imports = _extract_src_imports(py_file)

            for target, lineno, import_text in imports:
                # Self-imports are always fine
                if target == module_name:
                    continue
                if target not in allowed:
                    violations.append(
                        f"{rel_path}:{lineno} - "
                        f"{module_name} -> {target} "
                        f"(import {import_text})"
                    )

    return violations


def test_no_architecture_violations():
    """All cross-module imports must respect the layered architecture."""
    violations = _collect_violations()
    if violations:
        msg = (
            f"Found {len(violations)} architecture violation(s):\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\n\nSee docs/harness/ARCHITECTURE_RULES.md for allowed dependencies."
        )
        assert False, msg


def test_utils_has_no_src_imports():
    """src/utils/ must not import from any other src/ module."""
    utils_dir = SRC_DIR / "utils"
    if not utils_dir.exists():
        return

    violations = []
    for py_file in utils_dir.rglob("*.py"):
        imports = _extract_src_imports(py_file)
        for target, lineno, import_text in imports:
            if target != "utils":
                rel_path = py_file.relative_to(SRC_DIR.parent)
                violations.append(f"{rel_path}:{lineno} - import {import_text}")

    if violations:
        assert False, (
            f"src/utils/ must not import from other src/ modules:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


def test_no_circular_imports_at_module_level():
    """Check for circular dependency pairs between src/ subdirectories."""
    # Build a directed graph of module-level imports
    edges: dict[str, set[str]] = {}

    for subdir in sorted(SRC_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        module_name = subdir.name
        targets = set()
        for py_file in subdir.rglob("*.py"):
            for target, _, _ in _extract_src_imports(py_file):
                if target != module_name:
                    targets.add(target)
        if targets:
            edges[module_name] = targets

    # Check for bidirectional edges (A -> B and B -> A)
    circular = []
    checked = set()
    for module, deps in edges.items():
        for dep in deps:
            pair = tuple(sorted([module, dep]))
            if pair in checked:
                continue
            checked.add(pair)
            if dep in edges and module in edges[dep]:
                circular.append(f"{module} <-> {dep}")

    if circular:
        assert False, (
            f"Circular dependencies detected:\n"
            + "\n".join(f"  {c}" for c in circular)
        )


@pytest.mark.xfail(reason="Known: ui/ingest.py exceeds limit, tracked in docs/harness/ENTROPY_MANAGEMENT.md")
def test_file_size_limits():
    """Flag source files exceeding 600 lines."""
    oversized = []
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            line_count = sum(1 for _ in py_file.open(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
        if line_count > 600:
            rel_path = py_file.relative_to(SRC_DIR.parent)
            oversized.append(f"{rel_path}: {line_count} lines")

    if oversized:
        assert False, (
            f"Files exceeding 600 lines (split recommended):\n"
            + "\n".join(f"  {f}" for f in oversized)
        )
