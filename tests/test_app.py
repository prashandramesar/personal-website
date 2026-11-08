import io
import json
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from flask import render_template_string
from PIL import Image

import app as portfolio

FIXTURES = Path(__file__).parent / "fixtures"
UPLOADS = Path(portfolio.BASE_DIR) / "static" / "uploads"
PREDICT_ROUTES = ["/predict", "/api/dog/predict"]


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(portfolio.app) as test_client:
        yield test_client


def encode(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def golden(mode: str = "RGB") -> Image.Image:
    return Image.open(FIXTURES / "golden.jpeg").convert("RGB").convert(mode)


def post(client: TestClient, route: str, contents: bytes, name: str = "temp_image.jpg"):
    return client.post(route, files={"file": (name, contents, "image/jpeg")})


@pytest.mark.parametrize("path", list(portfolio.PAGES))
def test_pages_render(client: TestClient, path: str) -> None:
    response = client.get(path)
    html = response.text

    assert response.status_code == 200
    assert html.startswith("<!DOCTYPE html>")
    assert html.count("<div") == html.count("</div>")
    assert f"© {date.today().year} Prashand Ramesar" in html
    assert 'id="mobile-menu"' in html


@pytest.mark.parametrize("path", list(portfolio.PAGES))
def test_pages_have_share_tags(client: TestClient, path: str) -> None:
    html = client.get(path).text

    assert f'rel="canonical" href="{portfolio.SITE_URL}{path}"' in html
    assert 'property="og:title" content="' in html
    assert 'name="description" content="' in html
    assert "/static/img/og.png" in html


def test_kaai_stays_anonymous_and_attributes_the_cost_result(
    client: TestClient,
) -> None:
    html = client.get("/kaai").text

    assert "ANWB" not in html
    assert "Senior management" in html
    assert "operational cost" in html


def test_missing_page_shows_404_page_to_browsers(client: TestClient) -> None:
    response = client.get("/nope", headers={"Accept": "text/html"})

    assert response.status_code == 404
    assert "Page not found" in response.text


def test_missing_route_returns_json_to_api_clients(client: TestClient) -> None:
    response = client.get("/api/nope", headers={"Accept": "application/json"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_sitemap_and_robots(client: TestClient) -> None:
    sitemap = client.get("/sitemap.xml")

    assert sitemap.headers["content-type"].startswith("application/xml")
    for path in portfolio.PAGES:
        assert f"<loc>{portfolio.SITE_URL}{path}</loc>" in sitemap.text
    assert "/sitemap.xml" in client.get("/robots.txt").text


@pytest.mark.parametrize("route", PREDICT_ROUTES)
@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("golden.jpeg", "golden_retriever"),
        ("shihtzu.jpg", "Shih-Tzu"),
        ("dingo.jpg", "dingo"),
        ("dog-png-30.png", "golden_retriever"),
    ],
)
def test_predicts_sample_breeds(
    client: TestClient, route: str, filename: str, expected: str
) -> None:
    contents = (FIXTURES / filename).read_bytes()
    response = post(client, route, contents, filename)

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert expected in predictions[0]["breed"]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("n02099601-golden_retriever", "Golden Retriever"),
        ("n02086240-Shih-Tzu", "Shih-Tzu"),
        ("n02089078-black-and-tan_coonhound", "Black-and-tan Coonhound"),
        ("n02107908-Appenzeller", "Appenzeller"),
    ],
)
def test_format_breed_name(label: str, expected: str) -> None:
    assert portfolio.format_breed_name(label) == expected


def test_every_label_formats_without_error() -> None:
    labels = json.loads(portfolio.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    names = {portfolio.format_breed_name(label) for label in labels.values()}

    assert len(names) == 120


def test_response_keeps_fields_the_android_app_reads(client: TestClient) -> None:
    contents = (FIXTURES / "golden.jpeg").read_bytes()
    top = post(client, "/predict", contents).json()["predictions"][0]

    assert top["breed"] == "n02099601-golden_retriever"
    assert top["display_name"] == "Golden Retriever"
    assert isinstance(top["confidence"], float)


@pytest.mark.parametrize("mode", ["RGB", "L", "LA", "P", "RGBA", "I;16"])
def test_png_colour_modes_are_classified_correctly(
    client: TestClient, mode: str
) -> None:
    image = golden("L").convert("I;16") if mode == "I;16" else golden(mode)
    response = post(client, "/predict", encode(image), "dog.png")

    assert response.status_code == 200
    assert "golden_retriever" in response.json()["predictions"][0]["breed"]


def test_cmyk_jpeg_is_classified_correctly(client: TestClient) -> None:
    response = post(client, "/predict", encode(golden("CMYK"), "JPEG"))

    assert response.status_code == 200
    assert "golden_retriever" in response.json()["predictions"][0]["breed"]


def test_exif_rotation_is_applied() -> None:
    upright = golden().resize((400, 200))
    rotated = upright.rotate(90, expand=True)
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display.
    buffer = io.BytesIO()
    rotated.save(buffer, format="JPEG", quality=95, exif=exif)

    corrected = portfolio.load_upload(buffer.getvalue())
    unrotated = portfolio.load_upload(encode(upright, "JPEG"))
    ignored = portfolio.load_upload(encode(rotated, "JPEG"))

    assert abs(corrected - unrotated).mean() < abs(ignored - unrotated).mean()


@pytest.mark.parametrize("route", PREDICT_ROUTES)
@pytest.mark.parametrize(
    "contents",
    [b"not an image", b"", (FIXTURES / "golden.jpeg").read_bytes()[:500]],
    ids=["text", "empty", "truncated"],
)
def test_invalid_images_return_400(
    client: TestClient, route: str, contents: bytes
) -> None:
    assert post(client, route, contents).status_code == 400


@pytest.mark.parametrize("route", PREDICT_ROUTES)
def test_disallowed_format_with_allowed_extension_returns_400(
    client: TestClient, route: str
) -> None:
    contents = encode(golden().resize((50, 50)), "GIF")
    assert post(client, route, contents, "dog.jpg").status_code == 400


@pytest.mark.parametrize("route", PREDICT_ROUTES)
def test_missing_file_returns_400(client: TestClient, route: str) -> None:
    assert client.post(route).status_code == 400


@pytest.mark.parametrize("route", PREDICT_ROUTES)
def test_oversized_upload_returns_413(client: TestClient, route: str) -> None:
    contents = b"\0" * (portfolio.MAX_REQUEST_BYTES + 1)
    assert post(client, route, contents).status_code == 413


def test_file_just_over_limit_inside_request_allowance_returns_413(
    client: TestClient,
) -> None:
    contents = b"\0" * (portfolio.MAX_UPLOAD_BYTES + 1)
    assert post(client, "/predict", contents).status_code == 413


def test_chunked_upload_without_length_is_limited(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(18):
            yield b"\0" * (1024 * 1024)

    response = client.post(
        "/predict",
        content=chunks(),
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    assert response.status_code == 413


def test_decompression_bomb_returns_413(client: TestClient) -> None:
    contents = encode(Image.new("L", (8000, 8000)))
    assert len(contents) < portfolio.MAX_UPLOAD_BYTES
    assert post(client, "/predict", contents, "bomb.png").status_code == 413


def test_uploads_are_not_written_to_disk(client: TestClient) -> None:
    before = set(UPLOADS.iterdir()) if UPLOADS.exists() else set()
    post(client, "/predict", (FIXTURES / "golden.jpeg").read_bytes(), "new_photo.jpg")
    after = set(UPLOADS.iterdir()) if UPLOADS.exists() else set()

    assert after == before


def test_health_when_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_health_and_predict_report_missing_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(portfolio, "model", None)

    assert client.get("/health").status_code == 503
    contents = (FIXTURES / "golden.jpeg").read_bytes()
    assert post(client, "/predict", contents).status_code == 503


def test_cors_only_allows_the_site_origin(client: TestClient) -> None:
    allowed = client.get("/health", headers={"Origin": "https://prashand.nl"})
    other = client.get("/health", headers={"Origin": "https://example.com"})

    assert allowed.headers["access-control-allow-origin"] == "https://prashand.nl"
    assert "access-control-allow-origin" not in other.headers


def test_template_classes_are_in_the_compiled_stylesheet() -> None:
    """Fails if a template gained a class without rerunning build_css.py."""
    base_dir = Path(portfolio.BASE_DIR)
    stylesheet = (base_dir / "static" / "css" / "site.css").read_text(encoding="utf-8")
    compiled = {
        re.sub(r"\\(.)", r"\1", name)
        for name in re.findall(r"\.((?:\\.|[\w-])+)", stylesheet)
    }
    not_tailwind: set[str] = set()

    used: set[str] = set()
    for template in (base_dir / "templates").glob("*.html"):
        html = re.sub(r"{%.*?%}|{{.*?}}", " ", template.read_text("utf-8"))
        for classes in re.findall(r'class="([^"]*)"', html):
            used.update(classes.split())

    assert used - not_tailwind - compiled == set()


def test_large_jpeg_is_decoded_at_reduced_size() -> None:
    """Protects the 512 MB instance: a full decode here would need about 150 MB."""
    large = Image.new("RGB", (7000, 7000), "white")
    image = Image.open(io.BytesIO(encode(large, "JPEG")))

    portfolio.preprocess_image(image)

    assert max(image.size) < 1000


def test_large_jpeg_within_limit_is_accepted(client: TestClient) -> None:
    contents = encode(golden().resize((7000, 7000)), "JPEG")
    assert len(contents) < portfolio.MAX_UPLOAD_BYTES

    response = post(client, "/predict", contents)

    assert response.status_code == 200
    assert "golden_retriever" in response.json()["predictions"][0]["breed"]


def test_png_over_its_pixel_limit_returns_413(client: TestClient) -> None:
    contents = encode(Image.new("L", (5100, 5000)))
    assert post(client, "/predict", contents, "big.png").status_code == 413


COLOR_CLASS = re.compile(
    r"(?<![\w-])(?:hover:|focus:)?(?:bg|text|border|from|via|to|ring|divide|placeholder)-"
    r"(?:white|black|gray|red|yellow|green|blue|indigo|purple|pink)(?:-\d+)?(?![\w-])"
)
SHADOW_OR_GRADIENT = re.compile(
    r"(?<![\w-])(?:bg-gradient-to-\w+|shadow(?:-\w+)?)(?![\w-])"
)


def site_markup_files() -> list[Path]:
    base_dir = Path(portfolio.BASE_DIR)
    return [
        *(base_dir / "templates").glob("*.html"),
        *(base_dir / "static/js").glob("*.js"),
    ]


@pytest.mark.parametrize("path", site_markup_files(), ids=lambda path: path.name)
def test_templates_use_theme_colors_only(path: Path) -> None:
    """A hard-coded color looks wrong in one of the two themes."""
    text = path.read_text("utf-8")
    assert COLOR_CLASS.findall(text) == []
    assert SHADOW_OR_GRADIENT.findall(text) == []


THEME_CSS = (Path(portfolio.BASE_DIR) / "static" / "css" / "theme.css").read_text(
    "utf-8"
)


def theme_tokens(block_start: str) -> dict[str, str]:
    start = THEME_CSS.index(block_start)
    block = THEME_CSS[THEME_CSS.index("{", start) :]
    block = block[: block.index("}")]
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-f]{6})", block))


LIGHT = theme_tokens(":root {")
DARK = theme_tokens(':root[data-theme="dark"]')
SYSTEM_DARK = theme_tokens(':root:not([data-theme="light"])')
TEXT_PAIRS = [
    ("fg", "bg"),
    ("fg", "surface"),
    ("fg", "surface-raised"),
    ("muted", "bg"),
    ("muted", "surface"),
    ("muted", "surface-raised"),
    ("accent", "bg"),
    ("accent", "surface"),
    ("accent-hover", "bg"),
    ("on-accent", "accent"),
    ("on-accent", "accent-hover"),
    ("danger", "bg"),
    ("danger", "surface"),
    ("star", "bg"),
    ("star", "surface-raised"),
    ("on-success", "success"),
]


def luminance(color: str) -> float:
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(first: str, second: str) -> float:
    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
@pytest.mark.parametrize(("text", "background"), TEXT_PAIRS)
def test_theme_text_is_readable(
    theme: dict[str, str], text: str, background: str
) -> None:
    assert contrast(theme[text], theme[background]) >= 4.5


def test_both_ways_of_getting_dark_mode_match() -> None:
    assert DARK == SYSTEM_DARK
    assert set(DARK) == set(LIGHT)


def test_every_page_loads_the_theme_script_before_the_stylesheet(
    client: TestClient,
) -> None:
    html = client.get("/").text
    assert html.index("localStorage.getItem('theme')") < html.index("site.css")
    assert 'id="theme-toggle"' in html
    assert 'name="theme-color"' in html


def test_compiled_stylesheet_includes_the_current_theme() -> None:
    """Fails if theme.css changed without rerunning scripts/build_css.py."""
    compiled = (Path(portfolio.BASE_DIR) / "static" / "css" / "site.css").read_text(
        "utf-8"
    )
    assert THEME_CSS.replace("\r\n", "\n") in compiled.replace("\r\n", "\n")


def test_work_rows_show_a_photo_only_when_given_one() -> None:
    template = (
        "{% import 'work.html' as work %}"
        "{{ work.row('Plain', '/a', 'Text') }}"
        "{{ work.row('Photo', '/b', 'Text', image='b.jpg', image_alt='A dog') }}"
    )
    with portfolio.flask_app.app_context():
        html = render_template_string(template)
    assert html.count("<img") == 1
    assert 'src="/static/img/work/b.jpg" alt="A dog"' in html
    assert html.count("work-row-with-image") == 1
