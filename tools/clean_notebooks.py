"""Clear execution artifacts, or check that tracked notebooks are clean."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean(document):
    document = json.loads(json.dumps(document))
    document["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    for cell in document["cells"]:
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit with status 1 if execution artifacts remain.")
    args = parser.parse_args()
    changed = []
    for path in sorted(ROOT.glob("0*.ipynb")):
        original = json.loads(path.read_text(encoding="utf-8"))
        updated = clean(original)
        if original != updated:
            changed.append(path.name)
            if not args.check:
                path.write_text(json.dumps(updated, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.check and changed:
        print("Run python tools/clean_notebooks.py before committing: " + ", ".join(changed))
        return 1
    print("Notebook sources are clean." if args.check else f"Cleaned {len(changed)} notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
