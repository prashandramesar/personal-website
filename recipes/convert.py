"""Convert recipe quantities to metric and temperatures to Celsius.

All arithmetic happens here, so a language model never calculates a quantity.
Grams are only shown for a volume when the ingredient's density is known.
Anything uncertain gets a flag, so it can be checked before a recipe is saved.
"""

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

Region = Literal["US", "UK", "AU"]

FLAG_VOLUME_ONLY = "volume_only"
FLAG_ASSUMED_US = "assumed_us_measures"
FLAG_UNCLEAR_TEMPERATURE = "unclear_temperature"
FLAG_VAGUE_OVEN = "vague_oven_temperature"

DENSITIES_PATH = Path(__file__).with_name("densities.json")

GRAMS_PER_UNIT = {"oz": 28.35, "lb": 453.6, "g": 1.0, "kg": 1000.0, "ons": 100.0}
METRIC_ML_PER_UNIT = {"ml": 1.0, "cl": 10.0, "dl": 100.0, "l": 1000.0}
SPOON_ML = {"tbsp": 15.0, "tsp": 5.0}
# Measures that differ by country. Other regions use the US values.
US_ML_PER_UNIT = {
    "cup": 236.6,
    "fl oz": 29.6,
    "pint": 473.2,
    "quart": 946.4,
    "gallon": 3785.4,
}
REGION_ML_OVERRIDES: dict[str, dict[str, float]] = {
    "UK": {"pint": 568.3, "quart": 1136.5, "fl oz": 28.4, "gallon": 4546.1},
    "AU": {"cup": 250.0, "tbsp": 20.0},
}
STICK_OF_BUTTER_GRAMS = 113.0

UNIT_ALIASES = {
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "g": "g",
    "gr": "g",
    "gram": "g",
    "grams": "g",
    "gramme": "g",
    "grammes": "g",
    "kg": "kg",
    "kilo": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "ons": "ons",
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
    "cl": "cl",
    "dl": "dl",
    "l": "l",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "cup": "cup",
    "cups": "cup",
    "c": "cup",
    "tbsp": "tbsp",
    "tbs": "tbsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "el": "tbsp",
    "eetlepel": "tbsp",
    "eetlepels": "tbsp",
    "tsp": "tsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "tl": "tsp",
    "theelepel": "tsp",
    "theelepels": "tsp",
    "fl oz": "fl oz",
    "fl. oz": "fl oz",
    "fl. oz.": "fl oz",
    "floz": "fl oz",
    "fluid ounce": "fl oz",
    "fluid ounces": "fl oz",
    "pint": "pint",
    "pints": "pint",
    "quart": "quart",
    "quarts": "quart",
    "gallon": "gallon",
    "gallons": "gallon",
    "stick": "stick",
    "sticks": "stick",
}
# Non-metric units whose result depends on the country they came from.
REGION_SENSITIVE_UNITS = {"cup", "fl oz", "pint", "quart", "gallon"}

_METRIC_UNITS = (
    r"kg|kilograms?|g|gr|grams?|grammes?|ml|milliliters?|millilitres?"
    r"|cl|dl|l|liters?|litres?"
)
# The unit can appear on both ends of a range: "12g to 25g".
_METRIC_VALUE = (
    r"(?P<low>\d+(?:[.,]\d+)?)"
    r"(?:\s*(?:" + _METRIC_UNITS + r")?\s*(?:-|–|to)\s*(?P<high>\d+(?:[.,]\d+)?))?\s*"
    r"(?P<unit>" + _METRIC_UNITS + r")\b"
)
# A metric value the author gave next to another measure: "1 cup (120g)",
# "1/2 ounce (1 tablespoon; 15g)", "1 cup/240ml" or "100g/1 cup".
# The value must stand alone, so "(from a 400 g can)" is not read as the amount,
# and the slash can't follow a digit, so "1/2 l" is not read as 2 litres.
_AUTHOR_METRIC_PATTERNS = (
    re.compile(
        r"\(\s*(?:[^)]*(?:;\s*|,\s+))?(?:about|approx\.?|ca\.?|~|≈)?\s*"
        + _METRIC_VALUE
        + r"\s*[),;]",
        re.I,
    ),
    re.compile(r"(?<=[a-z.)\s])/\s*" + _METRIC_VALUE, re.I),
    re.compile(r"^\s*" + _METRIC_VALUE + r"\s*/", re.I),
)

_UNICODE_FRACTIONS = {
    "½": 0.5,
    "¼": 0.25,
    "¾": 0.75,
    "⅓": 1 / 3,
    "⅔": 2 / 3,
    "⅛": 0.125,
    "⅜": 0.375,
    "⅝": 0.625,
    "⅞": 0.875,
}
_WORD_NUMBERS = {
    "a": 1.0,
    "an": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "half a": 0.5,
    "half an": 0.5,
}
_FRACTION_CHARS = "".join(_UNICODE_FRACTIONS)
_NUMBER = (
    r"(?:\d+\s+\d+/\d+|\d+\s*[" + _FRACTION_CHARS + r"]|\d+/\d+|\d+(?:\.\d+)?"
    r"|[" + _FRACTION_CHARS + r"]|half an?|one|two|three|four|an?)"
)
# Only full words here: in running text, "t", "c" or "el" could be anything.
# An abbreviation's dot only counts when the sentence carries on after it, so
# "Add 8 oz." keeps its full stop.
_ABBREVIATION_DOT = r"(?:\.(?=\s+[a-z]))?"
_STEP_UNITS = (
    r"fl\.?\s?oz" + _ABBREVIATION_DOT + r"|fluid\s+ounces?|cups?|tablespoons?"
    r"|tbsp" + _ABBREVIATION_DOT + r"|tbs" + _ABBREVIATION_DOT + r"|teaspoons?"
    r"|tsp" + _ABBREVIATION_DOT + r"|eetlepels?|theelepels?|ounces?"
    r"|oz"
    + _ABBREVIATION_DOT
    + r"|pounds?|lbs?"
    + _ABBREVIATION_DOT
    + r"|pints?|quarts?|gallons?|sticks?"
)
_STEP_QUANTITY = re.compile(
    r"(?<![\w/.])(?P<low>" + _NUMBER + r")"
    r"(?:\s*(?:-|–|to)\s*(?P<high>" + _NUMBER + r"))?"
    r"[\s-]*(?P<unit>" + _STEP_UNITS + r")(?![\w-])"
    r"(?P<author>\s*\([^)]*\))?",
    re.I,
)
_STEP_UNIT_IN_TEXT = re.compile(r"\b(?:" + _STEP_UNITS + r")(?![\w-])", re.I)
# "Divide among 12 cups" means muffin cups or bowls, not a volume.
_CONTAINER_CONTEXT = re.compile(r"\b(among|between|into|fills?)\s+$", re.I)
_FOLLOWING_WORDS = re.compile(r"\s*(?:of\s+)?(?:the\s+)?([^.,;:()\n]*)")

_DEGREES = r"(?:\s*(?:°|º|degrees?|deg\.?))?"
_F_SUFFIX = _DEGREES + r"\s*(?:Fahrenheit|F)\b"
_C_SUFFIX = _DEGREES + r"\s*(?:Celsius|C)\b"
_RANGE = r"(?P<low>\d{2,3})(?:\s*(?:-|–|to)\s*(?P<high>\d{2,3}))?"
# "350°F (175°C)", "350F/175C": keep the author's Celsius and drop the rest.
_F_THEN_C = re.compile(
    r"\d{2,3}(?:\s*(?:-|–|to)\s*\d{2,3})?"
    + _F_SUFFIX
    + r"\s*[(/]\s*"
    + _RANGE
    + _C_SUFFIX
    + r"(?:\s*\))?",
    re.I,
)
# "175°C (350°F)", "180C/350F".
_C_THEN_F = re.compile(
    _RANGE
    + _C_SUFFIX
    + r"\s*[(/]\s*\d{2,3}(?:\s*(?:-|–|to)\s*\d{2,3})?"
    + _F_SUFFIX
    + r"(?:\s*\))?",
    re.I,
)
_FAHRENHEIT = re.compile(r"(?<![\d.])" + _RANGE + _F_SUFFIX, re.I)
# BBC style: "200C/180C fan/gas 6". Celsius is already there.
_GAS_AFTER_CELSIUS = re.compile(
    r"(\d{2,3}\s*°?\s*C(?:\s*/\s*\d{2,3}\s*°?\s*C\s*fan)?)\s*[/(,]\s*"
    r"gas(?:\s*mark|stand)?\s*[\d½¼/]+\)?",
    re.I,
)
_GAS_MARK = re.compile(
    r"\bgas(?:\s*mark|\s*regulo|stand)?\s*(?P<mark>\d|1/2|1/4|½|¼)(?!\d)", re.I
)
_DUTCH_DEGREES = re.compile(r"\b(\d{2,3})\s*graden\b(?:\s*Celsius)?", re.I)
_BARE_DEGREES = re.compile(
    r"(?<![\d.])(\d{2,3})\s*(?:°|º|degrees?\b)(?!\s*[CF]\b)", re.I
)
_VAGUE_OVEN = re.compile(
    r"\b(?:(?:very\s+)?(?:slow|moderate|medium|hot|warm|cool)\s+oven"
    r"|(?:matige|hete|warme)\s+oven)\b",
    re.I,
)
_OVEN_CONTEXT = re.compile(
    r"\b(?:ovens?|preheat|pre-heat|bake[sd]?|baking|roast|roasting|broil"
    r"|voorverwarm\w*|bakken|bak)\b",
    re.I,
)
GAS_MARK_CELSIUS = {
    "1/4": 110,
    "¼": 110,
    "1/2": 130,
    "½": 130,
    "1": 140,
    "2": 150,
    "3": 170,
    "4": 180,
    "5": 190,
    "6": 200,
    "7": 220,
    "8": 230,
    "9": 240,
}
# Ovens rarely go above this in Celsius, so a bare number over it must be
# Fahrenheit. Below it, "200 degrees" could be either.
_LOWEST_CERTAIN_FAHRENHEIT = 275
# Below this, a temperature in a baking sentence is the food's internal
# temperature, like "bake until it reads 206°F", so it keeps its precision.
_LOWEST_OVEN_FAHRENHEIT = 250
_ARTICLE_BEFORE_METRIC = re.compile(r"\b([Aa])n (?=\d+(?:\.\d+)? (?:g|kg|ml|l)\b)")


@dataclass(frozen=True)
class IngredientData:
    grams_per_us_cup: float | None


@dataclass(frozen=True)
class ConvertedIngredient:
    original_text: str
    name: str
    amount: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConvertedText:
    text: str
    flags: tuple[str, ...] = ()


@cache
def _ingredient_table() -> dict[str, IngredientData]:
    data = json.loads(DENSITIES_PATH.read_text(encoding="utf-8"))
    table = {name: IngredientData(None) for name in data["liquids"]}
    for name, entry in data["ingredients"].items():
        item = IngredientData(float(entry["grams_per_us_cup"]))
        table[name] = item
        for alias in entry["aliases"]:
            table[alias] = item
    return table


def find_ingredient(name: str | None) -> IngredientData | None:
    """Find an ingredient by name, or by the first known phrase within the name."""
    if not name:
        return None
    table = _ingredient_table()
    words = re.findall(r"[\w'’-]+", name.lower())[:6]
    for start in range(len(words)):
        for end in range(len(words), start, -1):
            match = table.get(" ".join(words[start:end]))
            if match:
                return match
    return None


def region_for_url(url: str | None) -> Region | None:
    """Return the region a site's domain shows, or None if it doesn't show one."""
    host = urlparse(url or "").hostname or ""
    if host.endswith(".uk"):
        return "UK"
    if host.endswith(".au"):
        return "AU"
    return None


def parse_number(text: str) -> float:
    """Parse '1 1/2', '1½', '3/4', '0.5', '½' or a word like 'half a'."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    if text in _WORD_NUMBERS:
        return _WORD_NUMBERS[text]
    total = 0.0
    for part in re.findall(r"\d+/\d+|\d+(?:\.\d+)?|[" + _FRACTION_CHARS + "]", text):
        if part in _UNICODE_FRACTIONS:
            total += _UNICODE_FRACTIONS[part]
        elif "/" in part:
            top, bottom = part.split("/")
            if int(bottom) == 0:
                raise ValueError(f"Invalid fraction: {part}")
            total += int(top) / int(bottom)
        else:
            total += float(part)
    return total


def _plain_number(value: float) -> str:
    return f"{round(value, 2):g}"


def _rounded_grams(grams: float) -> float:
    return round(grams * 2) / 2 if grams < 10 else float(round(grams))


def _rounded_ml(ml: float) -> float:
    if ml < 10:
        return round(ml * 2) / 2
    return float(round(ml)) if ml < 50 else float(5 * round(ml / 5))


def _format_metric(low: float, high: float | None, kind: Literal["g", "ml"]) -> str:
    """Format grams or millilitres, switching to kg or l from 1000 upwards."""
    rounder = _rounded_grams if kind == "g" else _rounded_ml
    large_unit = "kg" if kind == "g" else "l"
    top = high if high is not None else low
    if top >= 1000:
        low_text = _plain_number(low / 1000)
        high_text = _plain_number(high / 1000) if high is not None else None
        unit = large_unit
    else:
        low_text = _plain_number(rounder(low))
        high_text = _plain_number(rounder(high)) if high is not None else None
        unit = kind
    span = f"{low_text}-{high_text}" if high_text else low_text
    return f"{span} {unit}"


def _ml_per_unit(unit: str, region: Region | None) -> float:
    overrides = REGION_ML_OVERRIDES.get(region or "", {})
    if unit in overrides:
        return overrides[unit]
    if unit in SPOON_ML:
        return SPOON_ML[unit]
    return US_ML_PER_UNIT[unit]


def _convert_quantity(
    low: float,
    high: float | None,
    unit: str,
    ingredient: IngredientData | None,
    region: Region | None,
) -> tuple[str, tuple[str, ...]] | None:
    """Convert a known unit to display text and flags, or None if it isn't one."""
    flags: list[str] = []
    is_liquid = ingredient is not None and ingredient.grams_per_us_cup is None

    # Ounces of a liquid are fluid ounces, even when a recipe just says "oz".
    if unit == "oz" and is_liquid:
        unit = "fl oz"

    if unit == "stick":
        top = high * STICK_OF_BUTTER_GRAMS if high is not None else None
        return _format_metric(low * STICK_OF_BUTTER_GRAMS, top, "g"), ()
    if unit in GRAMS_PER_UNIT:
        scale = GRAMS_PER_UNIT[unit]
        top = high * scale if high is not None else None
        return _format_metric(low * scale, top, "g"), ()
    if unit in METRIC_ML_PER_UNIT:
        scale = METRIC_ML_PER_UNIT[unit]
        top = high * scale if high is not None else None
        return _format_metric(low * scale, top, "ml"), ()
    if unit not in SPOON_ML and unit not in US_ML_PER_UNIT:
        return None

    if unit in REGION_SENSITIVE_UNITS and region is None:
        flags.append(FLAG_ASSUMED_US)
    ml_each = _ml_per_unit(unit, region)
    ml_low = low * ml_each
    ml_high = high * ml_each if high is not None else None

    if unit in SPOON_ML:
        return _format_metric(ml_low, ml_high, "ml"), tuple(flags)

    if ingredient is not None and ingredient.grams_per_us_cup is not None:
        grams_per_ml = ingredient.grams_per_us_cup / US_ML_PER_UNIT["cup"]
        grams_high = ml_high * grams_per_ml if ml_high is not None else None
        text = _format_metric(ml_low * grams_per_ml, grams_high, "g")
        return f"≈ {text}", tuple(flags)

    if not is_liquid:
        flags.append(FLAG_VOLUME_ONLY)
    return _format_metric(ml_low, ml_high, "ml"), tuple(flags)


def _author_metric(text: str) -> str | None:
    """Return the metric value an author gave next to another measure, if any."""
    for pattern in _AUTHOR_METRIC_PATTERNS:
        match = pattern.search(text)
        if match:
            unit = UNIT_ALIASES[match["unit"].lower()]
            low = float(match["low"].replace(",", "."))
            high_text = match["high"]
            high = float(high_text.replace(",", ".")) if high_text else None
            converted = _convert_quantity(low, high, unit, None, None)
            if converted:
                return converted[0]
    return None


def _is_butter(*names: str | None) -> bool:
    return any(name and "butter" in name.lower() for name in names)


def convert_ingredient(
    original_text: str,
    name: str,
    quantity: float | None = None,
    quantity_max: float | None = None,
    unit: str | None = None,
    canonical_name: str | None = None,
    region: Region | None = None,
) -> ConvertedIngredient:
    """Convert one ingredient. A value the author gave in metric always wins."""
    if quantity is None:
        return ConvertedIngredient(original_text, name, "")

    unit_text = (unit or "").strip()
    # "T" is a tablespoon and "t" a teaspoon, so these two can't be lowercased.
    if unit_text in ("T", "Tb"):
        known_unit: str | None = "tbsp"
    elif unit_text == "t":
        known_unit = "tsp"
    else:
        known_unit = UNIT_ALIASES.get(unit_text.lower().rstrip("."))

    # Only butter comes in standard sticks. Cinnamon and celery don't.
    if known_unit == "stick" and not _is_butter(canonical_name, name):
        known_unit = None

    if known_unit is None:
        # Eggs, cloves, cans, "1 snufje": keep as written. A can's "(400 g)" is
        # per can, so it must not be read as the total.
        count = _plain_number(quantity)
        if quantity_max is not None:
            count = f"{count}-{_plain_number(quantity_max)}"
        return ConvertedIngredient(original_text, name, f"{count} {unit_text}".strip())

    ingredient = find_ingredient(canonical_name) or find_ingredient(name)
    converted = _convert_quantity(
        quantity, quantity_max, known_unit, ingredient, region
    )
    if converted is None:  # Every UNIT_ALIASES value is handled above.
        raise ValueError(f"Unhandled unit: {known_unit}")

    author_value = _author_metric(original_text)
    if author_value:
        return ConvertedIngredient(original_text, name, author_value)
    return ConvertedIngredient(original_text, name, converted[0], converted[1])


def _to_celsius(fahrenheit: int, nearest: int) -> int:
    return nearest * round((fahrenheit - 32) * 5 / 9 / nearest)


def _celsius_text(low: str, high: str | None) -> str:
    return f"{low}-{high}°C" if high else f"{low}°C"


def _convert_sentence_temperatures(sentence: str) -> tuple[str, set[str]]:
    flags: set[str] = set()
    is_oven = bool(_OVEN_CONTEXT.search(sentence))

    def nearest(fahrenheit_value: int) -> int:
        return 5 if is_oven and fahrenheit_value >= _LOWEST_OVEN_FAHRENHEIT else 1

    def author_celsius(match: re.Match[str]) -> str:
        return _celsius_text(match["low"], match["high"])

    def fahrenheit(match: re.Match[str]) -> str:
        low_f = int(match["low"])
        low = _to_celsius(low_f, nearest(low_f))
        if not match["high"]:
            return _celsius_text(str(low), None)
        high_f = int(match["high"])
        return _celsius_text(str(low), str(_to_celsius(high_f, nearest(high_f))))

    def bare_degrees(match: re.Match[str]) -> str:
        if not is_oven:  # Could be an angle: "turn the pan 90 degrees".
            return match[0]
        flags.add(FLAG_UNCLEAR_TEMPERATURE)
        value = int(match[1])
        if value >= _LOWEST_CERTAIN_FAHRENHEIT:
            return f"{_to_celsius(value, nearest(value))}°C"
        return match[0]

    def gas_mark(match: re.Match[str]) -> str:
        return f"{GAS_MARK_CELSIUS[match['mark']]}°C"

    sentence = _GAS_AFTER_CELSIUS.sub(r"\1", sentence)
    sentence = _F_THEN_C.sub(author_celsius, sentence)
    sentence = _C_THEN_F.sub(author_celsius, sentence)
    sentence = _FAHRENHEIT.sub(fahrenheit, sentence)
    sentence = _GAS_MARK.sub(gas_mark, sentence)
    sentence = _DUTCH_DEGREES.sub(r"\1°C", sentence)
    sentence = _BARE_DEGREES.sub(bare_degrees, sentence)
    if _VAGUE_OVEN.search(sentence):
        flags.add(FLAG_VAGUE_OVEN)
    return sentence, flags


def convert_temperatures(text: str) -> ConvertedText:
    """Rewrite temperatures in Celsius. Oven temperatures round to the nearest 5."""
    flags: set[str] = set()
    sentences = []
    # Keep whitespace between sentences, so the text is otherwise unchanged.
    for part in re.split(r"((?<=[.!?])\s+)", text):
        converted, part_flags = _convert_sentence_temperatures(part)
        sentences.append(converted)
        flags |= part_flags
    return ConvertedText("".join(sentences), tuple(sorted(flags)))


def _normalize_step_unit(unit: str) -> str:
    cleaned = re.sub(r"\s+", " ", unit.lower()).rstrip(".")
    return UNIT_ALIASES.get(cleaned, "fl oz" if cleaned.startswith("fl") else cleaned)


def convert_step(text: str, region: Region | None = None) -> ConvertedText:
    """Convert the quantities and temperatures inside one instruction step."""
    flags: set[str] = set()

    def quantity(match: re.Match[str]) -> str:
        before = match.string[: match.start()]
        after = match.string[match.end() :]
        following = _FOLLOWING_WORDS.match(after)
        following_text = following[1] if following else ""
        ingredient = find_ingredient(following_text)
        unit = _normalize_step_unit(match["unit"])

        # "Divide among 12 cups" is about containers, but "whisk into 1 cup milk"
        # names an ingredient, so it's still a quantity.
        if unit == "cup" and ingredient is None and _CONTAINER_CONTEXT.search(before):
            return match[0]
        if unit == "stick" and not _is_butter(following_text):
            return match[0]

        low = parse_number(match["low"])
        high = parse_number(match["high"]) if match["high"] else None
        converted = _convert_quantity(low, high, unit, ingredient, region)
        if converted is None:
            return match[0]

        author = match["author"] or ""
        author_value = _author_metric(author) if author else None
        if author_value:
            return author_value
        flags.update(converted[1])
        # A second measure like "(2 sticks)" would leave non-metric text behind.
        if author and _STEP_UNIT_IN_TEXT.search(author):
            author = ""
        return converted[0] + author

    converted_text = _STEP_QUANTITY.sub(quantity, text)
    if converted_text != text:
        converted_text = _ARTICLE_BEFORE_METRIC.sub(r"\1 ", converted_text)
    temperatures = convert_temperatures(converted_text)
    flags.update(temperatures.flags)
    return ConvertedText(temperatures.text, tuple(sorted(flags)))
