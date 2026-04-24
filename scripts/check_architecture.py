import os
import ast
import sys
from pathlib import Path

def check_no_illegal_imports_in_routers():
    """
    Ensure routers don't import vector_store, ingest, or raw SQLAlchemy ORM execution models,
    enforcing that all orchestration logic happens in the services/repositories.
    """
    routers_dir = Path("api/routers")
    if not routers_dir.exists():
        print("Skipping architecture check: api/routers not found.")
        return 0

    illegal_imports = {
        "vector_store": "Vector stores should be accessed via services.",
        "ingest": "Ingestion logic belongs in DocumentService.",
        "api.services.rag.get_embedding": "Embedding logic belongs in services.",
        "sqlalchemy.select": "SQL orchestration (select) should be in repositories.",
        "sqlalchemy.func": "SQL functions (func) should be in repositories."
    }

    errors = []

    for filepath in routers_dir.rglob("*.py"):
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for illegal, reason in illegal_imports.items():
                        if alias.name == illegal or alias.name.startswith(illegal + "."):
                            errors.append(f"{filepath}:{node.lineno} - Illegal import '{alias.name}'. {reason}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_import = f"{module}.{alias.name}" if module else alias.name
                    for illegal, reason in illegal_imports.items():
                        if full_import == illegal or module == illegal:
                            errors.append(f"{filepath}:{node.lineno} - Illegal import '{full_import}'. {reason}")

    if errors:
        print("FAIL: Architecture boundary violations found:")
        for error in errors:
            print(f"  {error}")
        return 1
    
    print("PASS: Architecture boundaries check passed! Routers are thin.")
    return 0

if __name__ == "__main__":
    sys.exit(check_no_illegal_imports_in_routers())
