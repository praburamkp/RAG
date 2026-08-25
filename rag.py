"""Small, dependency-free retrieval augmented generation toolkit."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}


@dataclass
class Chunk:
    source: str
    text: str
    terms: dict[str, float]


class LocalRAG:
    """A local RAG index using TF-IDF vectors and cosine similarity."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks or []
        self._idf: dict[str, float] = {}
        self._rebuild_idf()

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            token.lower()
            for token in TOKEN_RE.findall(text)
            if token.lower() not in STOPWORDS
        ]

    @classmethod
    def _term_frequency(cls, text: str) -> dict[str, float]:
        tokens = cls._tokens(text)
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        return {term: count / total for term, count in counts.items()}

    def _rebuild_idf(self) -> None:
        document_count = len(self.chunks)
        document_frequency = Counter(
            term for chunk in self.chunks for term in chunk.terms
        )
        self._idf = {
            term: math.log((1 + document_count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller")
        words = text.split()
        chunks: list[str] = []
        start = 0
        step = chunk_size - overlap
        while start < len(words):
            chunks.append(" ".join(words[start : start + chunk_size]))
            start += step
        return chunks

    def add_documents(
        self, documents: Iterable[tuple[str, str]], chunk_size: int = 180, overlap: int = 30
    ) -> int:
        added = 0
        for source, text in documents:
            for chunk_text in self.chunk_text(text, chunk_size, overlap):
                self.chunks.append(
                    Chunk(source, chunk_text, self._term_frequency(chunk_text))
                )
                added += 1
        self._rebuild_idf()
        return added

    def _vector(self, terms: dict[str, float]) -> dict[str, float]:
        return {term: frequency * self._idf.get(term, 0.0) for term, frequency in terms.items()}

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if top_k <= 0:
            return []
        query_vector = self._vector(self._term_frequency(query))
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        scored: list[tuple[Chunk, float]] = []
        for chunk in self.chunks:
            chunk_vector = self._vector(chunk.terms)
            denominator = query_norm * math.sqrt(
                sum(value * value for value in chunk_vector.values())
            )
            score = sum(
                query_vector.get(term, 0.0) * value
                for term, value in chunk_vector.items()
            ) / denominator if denominator else 0.0
            scored.append((chunk, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def answer(self, query: str, top_k: int = 3) -> str:
        results = [chunk.text for chunk, score in self.retrieve(query, top_k) if score > 0]
        if not results:
            return "I could not find an answer in the indexed documents."
        return "\n\n".join(results)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps([asdict(chunk) for chunk in self.chunks], indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalRAG":
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}. Run ingest first.")
        records = json.loads(path.read_text(encoding="utf-8"))
        return cls([Chunk(**record) for record in records])


def read_documents(folder: Path) -> list[tuple[str, str]]:
    documents = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            documents.append((str(path), path.read_text(encoding="utf-8")))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents and query a local RAG index")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="index .txt and .md files")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument("--index", type=Path, default=Path("rag_index.json"))
    ingest.add_argument("--chunk-size", type=int, default=180)
    ingest.add_argument("--overlap", type=int, default=30)
    query = subparsers.add_parser("query", help="retrieve context for a question")
    query.add_argument("question")
    query.add_argument("--index", type=Path, default=Path("rag_index.json"))
    query.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.command == "ingest":
        rag = LocalRAG()
        documents = read_documents(args.folder)
        count = rag.add_documents(documents, args.chunk_size, args.overlap)
        rag.save(args.index)
        print(f"Indexed {len(documents)} documents into {count} chunks: {args.index}")
    else:
        rag = LocalRAG.load(args.index)
        print(rag.answer(args.question, args.top_k))


if __name__ == "__main__":
    main()
