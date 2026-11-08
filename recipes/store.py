"""Saved recipes in SQLite, with FTS5 search. Must live on the Render disk."""

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS recipes (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    servings TEXT NOT NULL DEFAULT '',
    source_url TEXT UNIQUE,
    ingredients TEXT NOT NULL,
    steps TEXT NOT NULL,
    flags TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS recipe_search USING fts5(
    title, ingredients, steps, tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
EXPORT_VERSION = 1


class RestoreError(Exception):
    """An export that can't be restored."""


@dataclass(frozen=True)
class Recipe:
    id: int
    title: str
    servings: str
    source_url: str | None
    ingredients: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    flags: list[str]
    created_at: str


def _now() -> datetime:
    return datetime.now(UTC)


def _search_query(text: str) -> str:
    """Quote each word, so visitors' text can't be read as FTS5 query syntax."""
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    return " ".join(f'"{word}"*' for word in words)


class RecipeStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a connection for one transaction, committed if nothing raises."""
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _recipe(row: sqlite3.Row) -> Recipe:
        return Recipe(
            id=row["id"],
            title=row["title"],
            servings=row["servings"],
            source_url=row["source_url"],
            ingredients=json.loads(row["ingredients"]),
            steps=json.loads(row["steps"]),
            flags=json.loads(row["flags"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _write_search(
        connection: sqlite3.Connection,
        recipe_id: int,
        title: str,
        ingredients: list[dict[str, Any]],
        steps: list[dict[str, Any]],
    ) -> None:
        # Both the converted and original wording are searchable.
        ingredient_text = " ".join(
            f"{item.get('name', '')} {item.get('original', '')}" for item in ingredients
        )
        step_text = " ".join(str(step.get("text", "")) for step in steps)
        connection.execute(
            "INSERT INTO recipe_search (rowid, title, ingredients, steps) "
            "VALUES (?, ?, ?, ?)",
            (recipe_id, title, ingredient_text, step_text),
        )

    def save(
        self,
        title: str,
        servings: str,
        source_url: str | None,
        ingredients: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        flags: list[str] | None = None,
    ) -> int:
        """Save a recipe, or return the existing one if that link is already saved."""
        with self._connection() as connection:
            if source_url:
                existing = connection.execute(
                    "SELECT id FROM recipes WHERE source_url = ?", (source_url,)
                ).fetchone()
                if existing:
                    return int(existing["id"])
            cursor = connection.execute(
                "INSERT INTO recipes "
                "(title, servings, source_url, ingredients, steps, flags, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    title,
                    servings,
                    source_url,
                    json.dumps(ingredients, ensure_ascii=False),
                    json.dumps(steps, ensure_ascii=False),
                    json.dumps(flags or []),
                    _now().isoformat(),
                ),
            )
            recipe_id = int(cursor.lastrowid or 0)
            self._write_search(connection, recipe_id, title, ingredients, steps)
            return recipe_id

    def get(self, recipe_id: int) -> Recipe | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM recipes WHERE id = ?", (recipe_id,)
            ).fetchone()
        return self._recipe(row) if row else None

    def find_by_source(self, source_url: str) -> int | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id FROM recipes WHERE source_url = ?", (source_url,)
            ).fetchone()
        return int(row["id"]) if row else None

    def delete(self, recipe_id: int) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
            connection.execute(
                "DELETE FROM recipe_search WHERE rowid = ?", (recipe_id,)
            )

    def recent(self, limit: int = 100) -> list[Recipe]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM recipes ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._recipe(row) for row in rows]

    def search(self, text: str, limit: int = 100) -> list[Recipe]:
        query = _search_query(text)
        if not query:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT recipes.* FROM recipe_search "
                "JOIN recipes ON recipes.id = recipe_search.rowid "
                "WHERE recipe_search MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
        return [self._recipe(row) for row in rows]

    def record_import(self, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO imports (created_at, model, input_tokens, output_tokens) "
                "VALUES (?, ?, ?, ?)",
                (_now().isoformat(), model, input_tokens, output_tokens),
            )

    def imports_today(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM imports WHERE created_at >= ?",
                (_now().date().isoformat(),),
            ).fetchone()
        return int(row["total"])

    def usage_this_month(self) -> dict[str, int]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS imports, "
                "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
                "COALESCE(SUM(output_tokens), 0) AS output_tokens "
                "FROM imports WHERE created_at >= ?",
                (_now().strftime("%Y-%m-01"),),
            ).fetchone()
        return dict(row)

    def get_setting(self, key: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def export_json(self) -> str:
        """Export all recipes. Identical data gives identical text, for backups."""
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM recipes ORDER BY id").fetchall()
        recipes = [asdict(self._recipe(row)) for row in rows]
        return json.dumps(
            {"version": EXPORT_VERSION, "recipes": recipes},
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        )

    def restore_json(self, text: str) -> int:
        """Rebuild recipes from an export, into an empty database only."""
        try:
            data = json.loads(text)
            recipes = data["recipes"]
            if data["version"] != EXPORT_VERSION:
                raise RestoreError(f"Unsupported export version: {data['version']}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise RestoreError("That isn't a recipe export.") from e

        with self._connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
            if count:
                raise RestoreError("Refusing to restore over existing recipes.")
            try:
                for item in recipes:
                    connection.execute(
                        "INSERT INTO recipes (id, title, servings, source_url, "
                        "ingredients, steps, flags, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            item["id"],
                            item["title"],
                            item["servings"],
                            item["source_url"],
                            json.dumps(item["ingredients"], ensure_ascii=False),
                            json.dumps(item["steps"], ensure_ascii=False),
                            json.dumps(item["flags"]),
                            item["created_at"],
                        ),
                    )
                    self._write_search(
                        connection,
                        item["id"],
                        item["title"],
                        item["ingredients"],
                        item["steps"],
                    )
            except (KeyError, TypeError, sqlite3.IntegrityError) as e:
                raise RestoreError(f"That export is damaged: {e}") from e
        return len(recipes)
