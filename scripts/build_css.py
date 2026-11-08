"""Build static/css/site.css from Tailwind 2.2.19 and static/css/theme.css.

Keeps only the rules for classes the templates use, so visitors don't download
the full 3 MB file. Rerun this after adding a new Tailwind class to a template:

    python scripts/build_css.py
"""

import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_URL = (
    "https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css"
)
OUTPUT = ROOT / "static" / "css" / "site.css"
THEME = ROOT / "static" / "css" / "theme.css"
SCANNED_FILES = [*(ROOT / "templates").glob("*.html"), *(ROOT / "static").rglob("*.js")]
CLASS_IN_SELECTOR = re.compile(r"\.((?:\\.|[\w-])+)")


def used_tokens() -> set[str]:
    """Collect every word in the templates that could be a class name."""
    tokens: set[str] = set()
    for path in SCANNED_FILES:
        # Jinja tags can sit directly against a class name.
        text = re.sub(r"{%.*?%}|{{.*?}}", " ", path.read_text(encoding="utf-8"))
        tokens.update(re.split(r"[\s\"'`<>=]+", text))
    return tokens


def split_rules(css: str) -> list[tuple[str, str]]:
    """Split CSS into (prelude, body) pairs, matching nested braces."""
    rules = []
    start = position = 0
    while (opening := css.find("{", position)) != -1:
        depth = 1
        position = opening + 1
        while depth:
            character = css[position]
            depth += {"{": 1, "}": -1}.get(character, 0)
            position += 1
        rules.append((css[start:opening].strip(), css[opening + 1 : position - 1]))
        start = position
    return rules


def split_selectors(prelude: str) -> list[str]:
    """Split a selector list on commas that aren't inside parentheses."""
    selectors = []
    depth = 0
    current = ""
    for character in prelude:
        depth += {"(": 1, ")": -1}.get(character, 0)
        if character == "," and depth == 0:
            selectors.append(current)
            current = ""
        else:
            current += character
    return [*selectors, current]


def is_used(selector: str, tokens: set[str]) -> bool:
    classes = CLASS_IN_SELECTOR.findall(selector)
    return all(re.sub(r"\\(.)", r"\1", name) in tokens for name in classes)


def purge(css: str, tokens: set[str]) -> str:
    output = []
    for prelude, body in split_rules(css):
        if prelude.startswith("@keyframes"):
            output.append(f"{prelude}{{{body}}}")
        elif prelude.startswith("@"):
            if inner := purge(body, tokens):
                output.append(f"{prelude}{{{inner}}}")
        else:
            kept = [s for s in split_selectors(prelude) if is_used(s, tokens)]
            if kept:
                output.append(f"{','.join(kept)}{{{body}}}")
    return "".join(output)


def main() -> None:
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        source = response.read().decode("utf-8")
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # The theme comes last, so its tokens and components win over Tailwind.
    theme = THEME.read_text(encoding="utf-8")
    OUTPUT.write_text(purge(source, used_tokens()) + "\n" + theme, encoding="utf-8")
    print(f"{OUTPUT.name}: {len(source):,} -> {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
