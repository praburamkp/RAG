import tempfile
import unittest
from pathlib import Path

from RAG.rag import LocalRAG


class LocalRAGTests(unittest.TestCase):
    def test_retrieves_relevant_document(self):
        rag = LocalRAG()
        rag.add_documents([
            ("python.txt", "Python is useful for data science and automation."),
            ("cooking.txt", "A hot oven is useful for baking bread."),
        ])
        results = rag.retrieve("data science", top_k=1)
        self.assertEqual(results[0][0].source, "python.txt")
        self.assertGreater(results[0][1], 0)

    def test_index_round_trip(self):
        rag = LocalRAG()
        rag.add_documents([("notes.md", "RAG combines retrieval with generation.")])
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.json"
            rag.save(index)
            loaded = LocalRAG.load(index)
            self.assertIn("retrieval", loaded.answer("What does RAG combine?"))

    def test_missing_answer_is_explicit(self):
        rag = LocalRAG()
        rag.add_documents([("notes.txt", "The project uses Python.")])
        self.assertIn("could not find", rag.answer("What is the weather?"))


if __name__ == "__main__":
    unittest.main()
