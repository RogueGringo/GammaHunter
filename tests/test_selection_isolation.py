# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The selection package stays portable: relative imports inside it, and nothing beyond stdlib and torch."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "reachability_gen" / "selection"
ALLOWED_THIRD_PARTY = {"torch"}


def test_selection_package_imports_only_stdlib_torch_and_itself():
    files = sorted(PACKAGE.glob("*.py"))
    assert files
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY, (path.name, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import within the package
                    continue
                top = (node.module or "").split(".")[0]
                assert top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY, (path.name, node.module)
                assert top != "reachability_gen", (path.name, "absolute import of the host package")


def test_certificate_module_is_framework_free():
    tree = ast.parse((PACKAGE / "certificate.py").read_text(encoding="utf-8"))
    imported = {(a.name if isinstance(n, ast.Import) else n.module or "").split(".")[0]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert "torch" not in imported and all(m in sys.stdlib_module_names or m == "__future__" for m in imported if m)
