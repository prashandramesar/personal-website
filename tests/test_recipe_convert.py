"""Tests for recipe conversion. Pure functions, so no network or API calls."""

import json

import pytest

from recipes.convert import (
    DENSITIES_PATH,
    FLAG_ASSUMED_US,
    FLAG_UNCLEAR_TEMPERATURE,
    FLAG_VAGUE_OVEN,
    FLAG_VOLUME_ONLY,
    Region,
    convert_ingredient,
    convert_step,
    convert_temperatures,
    find_ingredient,
    parse_number,
    region_for_url,
)


@pytest.mark.parametrize(
    ("quantity", "unit", "name", "expected"),
    [
        (1, "cup", "butter", "≈ 227 g"),
        (2, "cups", "all-purpose flour", "≈ 250 g"),
        (0.5, "cup", "granulated sugar", "≈ 100 g"),
        (1, "cup", "packed brown sugar", "≈ 220 g"),
        (1, "cup", "unpacked brown sugar", "≈ 145 g"),
        (8, "oz", "cream cheese", "227 g"),
        (1, "lb", "potatoes", "454 g"),
        (3, "lb", "potatoes", "1.36 kg"),
        (2, "sticks", "butter", "226 g"),
        (1, "tbsp", "sugar", "15 ml"),
        (1, "tsp", "salt", "5 ml"),
        (0.5, "tsp", "salt", "2.5 ml"),
        (1, "T", "sugar", "15 ml"),
        (1, "t", "sugar", "5 ml"),
        (2, "eetlepels", "suiker", "30 ml"),
        (1, "cup", "milk", "235 ml"),
        (2, "fl oz", "lemon juice", "60 ml"),
        (8, "oz", "milk", "235 ml"),
    ],
)
def test_converts_units_with_known_region(
    quantity: float, unit: str, name: str, expected: str
) -> None:
    result = convert_ingredient(
        f"{quantity} {unit} {name}", name, quantity, unit=unit, region="US"
    )
    assert result.amount == expected
    assert result.flags == ()


@pytest.mark.parametrize(
    ("original", "quantity", "unit", "expected"),
    [
        ("300 gr bloem", 300, "gr", "300 g"),
        ("500 ml melk", 500, "ml", "500 ml"),
        ("1.5 kg aardappelen", 1.5, "kg", "1.5 kg"),
        ("2 dl room", 2, "dl", "200 ml"),
        ("25 cl bier", 25, "cl", "250 ml"),
        ("1 liter water", 1, "liter", "1 l"),
        ("1 ons kaas", 1, "ons", "100 g"),
    ],
)
def test_metric_units_stay_metric(
    original: str, quantity: float, unit: str, expected: str
) -> None:
    assert convert_ingredient(original, "x", quantity, unit=unit).amount == expected


@pytest.mark.parametrize(
    ("original", "quantity", "unit", "name", "expected"),
    [
        ("4 ounces plain Greek yogurt (1/2 cup; 115g)", 4, "ounces", "yogurt", "115 g"),
        ("4 medium bananas (20 ounces; 565g), ripe", 20, "ounces", "bananas", "565 g"),
        ("1 cup flour (120g)", 1, "cup", "flour", "120 g"),
        ("1 cup/240ml milk", 1, "cup", "milk", "240 ml"),
        ("100g/1 cup flour", 1, "cup", "flour", "100 g"),
        ("1 cup flour (120-130 g)", 1, "cup", "flour", "120-130 g"),
        ("2 tbsp oil (12g to 25g)", 2, "tbsp", "oil", "12-25 g"),
        ("1 cup sugar (1,5 dl)", 1, "cup", "sugar", "150 ml"),
    ],
)
def test_author_metric_wins_over_calculated_value(
    original: str, quantity: float, unit: str, name: str, expected: str
) -> None:
    result = convert_ingredient(original, name, quantity, unit=unit)
    assert result.amount == expected
    assert result.flags == ()


def test_metric_for_something_else_is_not_taken_as_the_amount() -> None:
    result = convert_ingredient(
        "1 cup flour, plus 30g for dusting", "flour", 1, unit="cup", region="US"
    )
    assert result.amount == "≈ 125 g"


@pytest.mark.parametrize(
    ("original", "quantity", "unit", "expected"),
    [
        ("2 eggs", 2, None, "2"),
        ("1 snufje zout", 1, "snufje", "1 snufje"),
        ("3 cloves garlic", 3, "cloves", "3 cloves"),
        ("2 cans (400 g) tomatoes", 2, "cans", "2 cans"),
        ("1-2 cloves garlic", 1, "cloves", "1-2 cloves"),
    ],
)
def test_counts_stay_as_written(
    original: str, quantity: float, unit: str | None, expected: str
) -> None:
    quantity_max = 2 if original.startswith("1-2") else None
    result = convert_ingredient(original, "x", quantity, quantity_max, unit)
    assert result.amount == expected
    assert result.flags == ()


@pytest.mark.parametrize(
    ("original", "quantity", "unit", "name", "expected"),
    [
        ("1 cup tomatoes (from a 400 g can)", 1, "cup", "tomatoes", "235 ml"),
        ("1/2 l milk", 0.5, "l", "milk", "500 ml"),
        ("1 1/2 kg potatoes", 1.5, "kg", "potatoes", "1.5 kg"),
        ("1 cup flour (about 120 g)", 1, "cup", "flour", "120 g"),
        ("1 cup butter (225g, softened)", 1, "cup", "butter", "225 g"),
    ],
)
def test_author_metric_is_only_used_when_it_is_the_amount(
    original: str, quantity: float, unit: str, name: str, expected: str
) -> None:
    result = convert_ingredient(original, name, quantity, unit=unit, region="US")
    assert result.amount == expected


@pytest.mark.parametrize(
    ("original", "name", "expected"),
    [
        ("2 sticks cinnamon", "cinnamon", "2 sticks"),
        ("1 stick celery", "celery", "1 stick"),
        ("2 sticks butter", "butter", "226 g"),
        ("1 stick unsalted butter", "unsalted butter", "113 g"),
    ],
)
def test_only_butter_converts_from_sticks(
    original: str, name: str, expected: str
) -> None:
    quantity = float(original.split()[0])
    unit = original.split()[1]
    assert convert_ingredient(original, name, quantity, unit=unit).amount == expected


def test_no_quantity_gives_no_amount() -> None:
    assert convert_ingredient("salt to taste", "salt").amount == ""


def test_unknown_ingredient_shows_ml_and_never_invents_grams() -> None:
    result = convert_ingredient(
        "1 cup dragon fruit", "dragon fruit", 1, unit="cup", region="US"
    )
    assert result.amount == "235 ml"
    assert FLAG_VOLUME_ONLY in result.flags
    assert "g" not in result.amount.split()


def test_liquid_is_not_flagged_as_volume_only() -> None:
    result = convert_ingredient("1 cup water", "water", 1, unit="cup", region="US")
    assert result.flags == ()


def test_canonical_name_finds_a_dutch_ingredient() -> None:
    result = convert_ingredient(
        "1 cup bloem",
        "bloem",
        1,
        unit="cup",
        canonical_name="all-purpose flour",
        region="US",
    )
    assert result.amount == "≈ 125 g"


def test_ranges_convert_both_ends() -> None:
    result = convert_ingredient("1-2 cups flour", "flour", 1, 2, "cups", region="US")
    assert result.amount == "≈ 125-250 g"


def test_unknown_region_assumes_us_and_flags_it() -> None:
    result = convert_ingredient("1 cup flour", "flour", 1, unit="cup")
    assert result.amount == "≈ 125 g"
    assert FLAG_ASSUMED_US in result.flags


def test_unknown_region_does_not_flag_weights_or_spoons() -> None:
    assert convert_ingredient("1 lb beef", "beef", 1, unit="lb").flags == ()
    assert convert_ingredient("1 tbsp oil", "oil", 1, unit="tbsp").flags == ()


@pytest.mark.parametrize(
    ("region", "quantity", "unit", "name", "expected"),
    [
        ("UK", 1, "pint", "milk", "570 ml"),
        ("US", 1, "pint", "milk", "475 ml"),
        ("AU", 1, "cup", "milk", "250 ml"),
        ("AU", 1, "tbsp", "milk", "20 ml"),
        ("AU", 1, "cup", "flour", "≈ 132 g"),
    ],
)
def test_regional_measures(
    region: Region, quantity: float, unit: str, name: str, expected: str
) -> None:
    result = convert_ingredient(
        f"{quantity} {unit} {name}", name, quantity, unit=unit, region=region
    )
    assert result.amount == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.bbcgoodfood.com/recipes/x", None),
        ("https://www.example.co.uk/x", "UK"),
        ("https://www.example.com.au/x", "AU"),
        ("https://www.leukerecepten.nl/x", None),
        (None, None),
    ],
)
def test_region_for_url(url: str | None, expected: str | None) -> None:
    assert region_for_url(url) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 1/2", 1.5),
        ("1½", 1.5),
        ("3/4", 0.75),
        ("0.5", 0.5),
        ("½", 0.5),
        ("half a", 0.5),
        ("one", 1.0),
    ],
)
def test_parse_number(text: str, expected: float) -> None:
    assert parse_number(text) == expected


def test_parse_number_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError):
        parse_number("1/0")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Preheat oven to 350°F.", "Preheat oven to 175°C."),
        ("Bake at 350 F for 20 minutes.", "Bake at 175°C for 20 minutes."),
        ("Bake at 400f.", "Bake at 205°C."),
        ("Preheat the oven to 350 degrees F.", "Preheat the oven to 175°C."),
        ("Preheat to 350 degrees Fahrenheit.", "Preheat to 175°C."),
        ("Roast at 375-400°F.", "Roast at 190-205°C."),
        ("Bake at 350°F (175°C).", "Bake at 175°C."),
        ("Bake at 350 degrees F (175 degrees C).", "Bake at 175°C."),
        ("Bake at 180°C (350°F).", "Bake at 180°C."),
        ("Bake at 180C/350F.", "Bake at 180°C."),
        ("Heat oven to 200C/180C fan/gas 6.", "Heat oven to 200C/180C fan."),
        ("Bake at gas mark 4.", "Bake at 180°C."),
        ("Bake at gasstand 6.", "Bake at 200°C."),
        ("Verwarm de oven voor op 180 graden.", "Verwarm de oven voor op 180°C."),
        ("Bake until the centre reads 206°F.", "Bake until the centre reads 97°C."),
        ("Cook until it reads 206°F inside.", "Cook until it reads 97°C inside."),
        ("Warm the milk to 110°F.", "Warm the milk to 43°C."),
        ("Preheat to 200°C.", "Preheat to 200°C."),
    ],
)
def test_temperatures(text: str, expected: str) -> None:
    assert convert_temperatures(text).text == expected


def test_only_the_oven_sentence_rounds_to_five() -> None:
    result = convert_temperatures("Bake at 350°F. Cook until it reads 206°F.")
    assert result.text == "Bake at 175°C. Cook until it reads 97°C."


def test_bare_high_number_in_oven_context_is_fahrenheit_and_flagged() -> None:
    result = convert_temperatures("Preheat the oven to 350 degrees.")
    assert result.text == "Preheat the oven to 175°C."
    assert FLAG_UNCLEAR_TEMPERATURE in result.flags


def test_bare_low_number_in_oven_context_stays_and_is_flagged() -> None:
    result = convert_temperatures("Preheat the oven to 200 degrees.")
    assert result.text == "Preheat the oven to 200 degrees."
    assert FLAG_UNCLEAR_TEMPERATURE in result.flags


def test_bare_degrees_outside_an_oven_is_left_alone() -> None:
    result = convert_temperatures("Turn the pan 90 degrees.")
    assert result.text == "Turn the pan 90 degrees."
    assert result.flags == ()


def test_vague_oven_stays_as_written_and_is_flagged() -> None:
    result = convert_temperatures("Bake in a moderate oven.")
    assert result.text == "Bake in a moderate oven."
    assert FLAG_VAGUE_OVEN in result.flags


def test_text_without_temperatures_is_unchanged() -> None:
    text = "Fold gently.  Rest for 10 minutes!\nServe warm."
    assert convert_temperatures(text) == convert_temperatures(text)
    assert convert_temperatures(text).text == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Add 1 cup flour and mix.", "Add ≈ 125 g flour and mix."),
        ("Add 1 cup of the flour.", "Add ≈ 125 g of the flour."),
        ("Stir in 1 1/2 cups sugar.", "Stir in ≈ 300 g sugar."),
        ("Stir in ½ cup sugar.", "Stir in ≈ 100 g sugar."),
        ("Add a cup of milk.", "Add 235 ml of milk."),
        ("Add 2 tablespoons butter.", "Add 30 ml butter."),
        ("Pour in 1-2 cups water.", "Pour in 235-475 ml water."),
        ("Add 8 oz cream cheese.", "Add 227 g cream cheese."),
        ("Use an 8-ounce block.", "Use a 227 g block."),
        ("Add 8 oz.", "Add 227 g."),
        ("Add 8 oz. cream cheese.", "Add 227 g cream cheese."),
        ("Whisk into 1 cup milk.", "Whisk into 235 ml milk."),
        ("Add 1 cup (2 sticks) butter.", "Add ≈ 227 g butter."),
        ("Add 1 cup (see note) flour.", "Add ≈ 125 g (see note) flour."),
        ("Add a stick of butter.", "Add 113 g of butter."),
        ("Add 1/2 cup (120 g) milk.", "Add 120 g milk."),
    ],
)
def test_step_quantities(text: str, expected: str) -> None:
    assert convert_step(text, region="US").text == expected


def test_step_converts_temperature_and_quantity_together() -> None:
    result = convert_step("Add 1 cup flour, then bake at 350°F.", region="US")
    assert result.text == "Add ≈ 125 g flour, then bake at 175°C."


def test_step_collects_flags() -> None:
    result = convert_step("Add 1 cup dragon fruit, then bake at 350 degrees.")
    assert result.text == "Add 235 ml dragon fruit, then bake at 175°C."
    assert set(result.flags) == {
        FLAG_ASSUMED_US,
        FLAG_VOLUME_ONLY,
        FLAG_UNCLEAR_TEMPERATURE,
    }


@pytest.mark.parametrize(
    "text",
    [
        "Divide the batter among 12 cups.",
        "Mix everything well.",
        "Cook for 5 minutes.",
        "Roll out to 2 cm thick.",
        "Add 200 g flour.",
        "Add 2 sticks cinnamon.",
        "Add an 8 minute rest.",
    ],
)
def test_steps_without_convertible_quantities_are_unchanged(text: str) -> None:
    assert convert_step(text, region="US").text == text


def test_density_table_is_valid_and_has_a_source_for_every_value() -> None:
    data = json.loads(DENSITIES_PATH.read_text(encoding="utf-8"))
    names: list[str] = list(data["liquids"])
    for name, entry in data["ingredients"].items():
        assert entry["grams_per_us_cup"] > 0, name
        assert entry["fdc_id"] > 0, name
        assert entry["usda_portion"], name
        names += [name, *entry["aliases"]]
    assert len(names) == len(set(names)), "A name maps to two ingredients"


def test_find_ingredient_uses_a_phrase_within_a_longer_name() -> None:
    butter = find_ingredient("butter")
    assert butter is not None
    assert find_ingredient("softened butter") == butter
    assert find_ingredient("butter, softened") == butter
    assert find_ingredient("dragon fruit") is None
    assert find_ingredient(None) is None
