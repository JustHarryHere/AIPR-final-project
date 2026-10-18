import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
from schemas import Action, parse_action
from rag.text import tokenize
from mcp_server.server import calc_required_score, calc_workload
from rag.chunking import Document, chunk_by_heading
from rag.retriever import Retriever, resolve_versions


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
        res = calc_workload(hours_per_week=4.0, from_week=1, to_week=6)
        self.assertTrue(res["ok"])
        self.assertEqual(res["data"]["total_hours"], 24.0)

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

    def test_chunking_heading(self):
        doc = Document(content="# Section 1\nContent 1\n## Sub 2\nContent 2", doc_id="test_doc", title="Test")
        chunks = chunk_by_heading(doc)
        self.assertGreaterEqual(len(chunks), 1)

    def test_retriever_loads_all_md_files(self):
        """Kiểm tra Retriever tự động quét và đọc tất cả các file .md trong data/docs/."""
        retriever = Retriever()
        self.assertGreaterEqual(len(retriever.chunks), 20)

        # Kiểm tra sự hiện diện của từng file bài giảng tuần 1 đến tuần 6
        sources = {c.metadata.get("file") for c in retriever.chunks}
        for w in range(1, 7):
            expected_file = f"w{w}_lecture_notes.md"
            self.assertIn(expected_file, sources, f"Thiếu file bài giảng {expected_file} trong retriever!")

    def test_retriever_search_lecture_notes(self):
        """Kiểm tra tìm kiếm nội dung thực tế trong các file markdown bài giảng."""
        retriever = Retriever()

        # 1. Tra cứu kiến thức Tuần 3 (RAG Foundations, DocumentStore, Cosine Similarity)
        res_w3 = retriever.search("DocumentStore cosine similarity embeddings", top_k=3)
        self.assertGreater(len(res_w3), 0)
        self.assertGreater(res_w3[0]["score"], 0.20)
        self.assertIn("w3_lecture_notes.md", [r["source"] for r in res_w3])

        # 2. Tra cứu kiến thức Tuần 6 (ReAct Agent loop, Thought, Action, Observation)
        res_w6 = retriever.search("ReAct Thought Action Observation Agent loop", top_k=3)
        self.assertGreater(len(res_w6), 0)
        self.assertGreater(res_w6[0]["score"], 0.20)
        self.assertIn("w6_lecture_notes.md", [r["source"] for r in res_w6])

    def test_resolve_versions_logic(self):
        """Kiểm tra giải quyết xung đột phiên bản: ưu tiên official v2.0 thay vì draft v1.0."""
        mock_meta = [
            {"doc_id": "doc_v1", "course_id": "AIPR", "doc_type": "policy", "version": "1.0", "status": "draft", "date": "2026-09-01"},
            {"doc_id": "doc_v2", "course_id": "AIPR", "doc_type": "policy", "version": "2.0", "status": "official", "date": "2026-10-01"},
        ]
        active, warnings = resolve_versions(mock_meta)
        self.assertIn("doc_v2", active)
        self.assertNotIn("doc_v1", active)

    def test_retriever_auto_detect_unregistered_md(self):
        """Kiểm tra Retriever tự động đọc bất kỳ file .md mới nào mà không cần khai báo trước vào metadata.json."""
        docs_dir = ROOT / "data" / "docs"
        test_file = docs_dir / "_temp_test_auto_discovery.md"
        test_file.write_text("## Test Section\nNội dung bài học thử nghiệm tự động quét file markdown.", encoding="utf-8")
        try:
            r = Retriever()
            res = r.search("thử nghiệm tự động quét file markdown", top_k=2)
            self.assertGreater(len(res), 0)
            self.assertEqual(res[0]["source"], "_temp_test_auto_discovery.md")
        finally:
            test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
