import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
from schemas import Action, parse_action
from rag.text import tokenize
from rag.calc import calc_required_score, calc_workload
from rag.chunking import Document, chunk_by_heading


class TestStudyAssistantUnit(unittest.TestCase):
    def test_tokenize(self):
        terms = tokenize("AIPR Tuần 1: Text Processing!")
        self.assertIn("aipr", terms)
        self.assertIn("tuần", terms)
        self.assertIn("text", terms)

    def test_calc_required_score_reachable(self):
        res = calc_required_score(midterm=7.0, w_mid=0.4, w_final=0.6, target=8.0)
        self.assertTrue(res["ok"])
        self.assertEqual(res["data"]["required_final_score"], 8.67)
        self.assertTrue(res["data"]["reachable"])

    def test_calc_required_score_invalid(self):
        res = calc_required_score(midterm=-1, w_mid=0.4, w_final=0.6, target=8.0)
        self.assertFalse(res["ok"])

    def test_calc_workload(self):
        res = calc_workload(hours_per_week=4.5, from_week=1, to_week=4)
        self.assertTrue(res["ok"])
        self.assertEqual(res["data"]["total_hours"], 18.0)

    def test_parse_action_tool(self):
        raw = '{"type": "tool", "tool": "search_lesson_plan", "args": {"query": "tuần 1"}}'
        act = parse_action(raw)
        self.assertIsNotNone(act)
        self.assertEqual(act.type, "tool")
        self.assertEqual(act.tool, "search_lesson_plan")

    def test_parse_action_final(self):
        raw = '{"type": "final", "answer": "Hoàn thành."}'
        act = parse_action(raw)
        self.assertIsNotNone(act)
        self.assertEqual(act.type, "final")
        self.assertEqual(act.answer, "Hoàn thành.")

    def test_chunking(self):
        doc = Document(content="# Section 1\nContent 1\n## Sub 2\nContent 2", doc_id="test_doc", title="Test")
        chunks = chunk_by_heading(doc)
        self.assertGreaterEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
