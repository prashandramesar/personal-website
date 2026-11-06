import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
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


@pytest.mark.parametrize(
    "path",
    ["/", "/projects", "/dog_classifier", "/about", "/android_app", "/this_website"],
)
def test_pages_render(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200


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


def test_response_keeps_fields_the_android_app_reads(client: TestClient) -> None:
    contents = (FIXTURES / "golden.jpeg").read_bytes()
    top = post(client, "/predict", contents).json()["predictions"][0]

    assert top["breed"] == "n02099601-golden_retriever"
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
