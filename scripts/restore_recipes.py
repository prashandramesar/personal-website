"""Rebuild the recipe database from a JSON export or backup.

Only writes to an empty database, so it can't overwrite saved recipes:

    python scripts/restore_recipes.py recipes.json /var/data/recipes.db
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipes.store import RecipeStore, RestoreError  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    export_path, db_path = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        count = RecipeStore(db_path).restore_json(
            export_path.read_text(encoding="utf-8")
        )
    except (OSError, RestoreError) as e:
        print(f"Restore failed: {e}")
        return 1
    print(f"Restored {count} recipes to {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
