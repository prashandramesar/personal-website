"""Answers questions about Prashand's work from the approved text in knowledge/.

The free version finds and returns existing passages using BM25 keyword search.
It generates nothing and calls no paid API. A later LLM version only needs to
implement AnswerProvider.
"""

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
MAX_QUESTION_LENGTH = 300
MAX_PASSAGES = 2
# The share of a question's meaning that must be found in the knowledge base.
# Rare words count for more, so "work at Google" fails even though "work" matches.
MIN_QUESTION_COVERAGE = 0.5
# Also return a second passage only if it scores close to the best one.
MIN_RELATIVE_SCORE = 0.6
BM25_K1 = 1.5
BM25_B = 0.75

SUGGESTED_QUESTIONS = [
    "What is KAAI?",
    "What was your role on KAAI?",
    "Why was NBA delayed?",
    "How was the dog breed model built?",
]

STOPWORDS = frozenset(
    """a about after all also am an and any are as at be been but by can could did do
    does for from get give had has have he her his how i if in into is it its just me
    more my of on or our she so some tell than that the their them they this to us was
    we were what when where which who why will with would you your please know there
    kind find make made use using take took""".split()
)
# Common ways of asking for something the knowledge base words differently.
SYNONYMS = {
    "accomplishment": "result",
    "achievement": "result",
    "achieve": "result",
    "background": "experience",
    "built": "build",
    "college": "university",
    "degree": "education",
    "delayed": "delay",
    "employer": "job",
    "experience": "experience",
    "money": "cost",
    "postpone": "delay",
    "postponed": "delay",
    "reach": "contact",
    "resume": "resume",
    "responsibility": "accountable",
    "responsible": "accountable",
    "retro": "retrospective",
    "save": "cost",
    "study": "education",
    "studied": "education",
    "website": "website",
    "work": "job",
    "worked": "job",
}


def stem(word: str) -> str:
    """Reduce related forms to one, such as handling and handle, or built and build."""
    for suffix in ("ion", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and not word.endswith("ss"):
            if len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word not in STOPWORDS and len(word) > 1:
            tokens.append(stem(SYNONYMS.get(word, word)))
    return tokens


@dataclass(frozen=True)
class Passage:
    title: str
    text: str
    source_url: str
    source_label: str


@dataclass(frozen=True)
class Answer:
    answered: bool
    passages: list[Passage]
    suggestions: list[str]
    provider: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerProvider(Protocol):
    def answer(self, question: str) -> Answer: ...


def load_passages(directory: Path = KNOWLEDGE_DIR) -> list[Passage]:
    """Read '# Page', 'Source: /url' and '## Passage title' from Markdown files."""
    passages = []
    for path in sorted(directory.glob("*.md")):
        page = source = title = ""
        lines: list[str] = []

        def flush() -> None:
            if title and lines:
                text = " ".join(lines).strip()
                passages.append(Passage(title, text, source, page))

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                flush()
                title, lines = line[3:], []
            elif line.startswith("# "):
                flush()
                page, title, lines = line[2:], "", []
            elif line.startswith("Source: "):
                source = line.removeprefix("Source: ")
            elif line:
                lines.append(line)
        flush()
    return passages


class LocalRetrievalProvider:
    """BM25 search over the knowledge base. Returns existing text word for word."""

    name = "local-retrieval"

    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        # Titles count too, as they say what a passage is about.
        self.documents = [
            Counter(tokenize(f"{p.title} {p.title} {p.text}")) for p in passages
        ]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(document.keys())
        count = len(passages)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.unknown_idf = math.log(1 + (count + 0.5) / 0.5)

    def score(self, tokens: list[str], index: int) -> float:
        document = self.documents[index]
        length_ratio = self.lengths[index] / self.average_length
        total = 0.0
        for token in tokens:
            frequency = document.get(token, 0)
            if frequency:
                total += (
                    self.idf[token]
                    * frequency
                    * (BM25_K1 + 1)
                    / (frequency + BM25_K1 * (1 - BM25_B + BM25_B * length_ratio))
                )
        return total

    def coverage(self, tokens: list[str]) -> float:
        weights = {t: self.idf.get(t, self.unknown_idf) for t in set(tokens)}
        known = sum(weight for token, weight in weights.items() if token in self.idf)
        return known / sum(weights.values())

    def answer(self, question: str) -> Answer:
        tokens = tokenize(question)
        if tokens and self.coverage(tokens) >= MIN_QUESTION_COVERAGE:
            scores = [self.score(tokens, i) for i in range(len(self.passages))]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            best = scores[ranked[0]]
            if best > 0:
                matches = [
                    self.passages[i]
                    for i in ranked[:MAX_PASSAGES]
                    if scores[i] >= best * MIN_RELATIVE_SCORE
                ]
                return Answer(True, matches, [], self.name)
        return Answer(False, [], SUGGESTED_QUESTIONS, self.name)


@lru_cache(maxsize=1)
def get_provider() -> AnswerProvider:
    return LocalRetrievalProvider(load_passages())
