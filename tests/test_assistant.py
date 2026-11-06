import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as portfolio
import assistant

CASES = json.loads(
    (Path(__file__).parent.parent / "evaluation" / "assistant_cases.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(portfolio.app) as test_client:
        yield test_client


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTANT_ENABLED", "true")


@pytest.mark.parametrize("case", CASES["answerable"], ids=lambda case: case["question"])
def test_eval_finds_the_right_passage_first(case: dict[str, str]) -> None:
    answer = assistant.get_provider().answer(case["question"])

    assert answer.answered
    assert answer.passages[0].title == case["title"]


@pytest.mark.parametrize("question", CASES["unanswerable"])
def test_eval_says_it_does_not_know(question: str) -> None:
    answer = assistant.get_provider().answer(question)

    assert not answer.answered
    assert answer.passages == []
    assert answer.suggestions == assistant.SUGGESTED_QUESTIONS


def test_suggested_questions_can_all_be_answered() -> None:
    for question in assistant.SUGGESTED_QUESTIONS:
        assert assistant.get_provider().answer(question).answered


def test_every_passage_links_to_a_real_page() -> None:
    passages = assistant.load_passages()

    assert len(passages) > 10
    for passage in passages:
        assert passage.source_url in portfolio.PAGES
        assert passage.text


def test_answers_are_existing_text_word_for_word() -> None:
    knowledge = " ".join(
        " ".join(path.read_text(encoding="utf-8").split())
        for path in assistant.KNOWLEDGE_DIR.glob("*.md")
    )
    answer = assistant.get_provider().answer("What is KAAI?")

    for passage in answer.passages:
        assert passage.text in knowledge


def test_kaai_knowledge_stays_anonymous() -> None:
    assert "ANWB" not in (assistant.KNOWLEDGE_DIR / "kaai.md").read_text("utf-8")


def test_hidden_by_default(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("ASSISTANT_ENABLED", raising=False)

    assert client.get("/assistant").status_code == 404
    response = client.post("/api/assistant/ask", json={"question": "What is KAAI?"})
    assert response.status_code == 404
    assert 'href="/assistant"' not in client.get("/").text
    assert "/assistant" not in client.get("/sitemap.xml").text


@pytest.mark.usefixtures("enabled")
def test_page_when_enabled(client: TestClient) -> None:
    html = client.get("/assistant").text

    assert "finds existing text" in html
    assert all(
        question.replace("'", "&#39;") in html
        for question in assistant.SUGGESTED_QUESTIONS
    )
    assert 'href="/assistant"' in client.get("/").text


@pytest.mark.usefixtures("enabled")
def test_ask_returns_cited_passages(client: TestClient) -> None:
    response = client.post("/api/assistant/ask", json={"question": "What is KAAI?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["provider"] == "local-retrieval"
    assert body["passages"][0]["source_url"] == "/kaai"


@pytest.mark.usefixtures("enabled")
def test_ask_declines_with_suggestions(client: TestClient) -> None:
    body = client.post(
        "/api/assistant/ask", json={"question": "What is your salary?"}
    ).json()

    assert body["answered"] is False
    assert body["suggestions"] == assistant.SUGGESTED_QUESTIONS


@pytest.mark.usefixtures("enabled")
@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "x" * (assistant.MAX_QUESTION_LENGTH + 1)}],
    ids=["missing", "empty", "too-long"],
)
def test_ask_rejects_invalid_questions(client: TestClient, payload: dict) -> None:
    assert client.post("/api/assistant/ask", json=payload).status_code == 422
