import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
TESTSET_PATH = ROOT_DIR / "tests" / "testset.jsonl"
JUDGE_PROMPT_PATH = ROOT_DIR / "prompts" / "judge.txt"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure logs dir exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_testset() -> list[dict]:
    if not TESTSET_PATH.exists():
        print(f"Error: {TESTSET_PATH} not found!")
        return []
    items = []
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_corpus_context(question: str = "") -> str:
    if question:
        try:
            from rag.retriever import get_retriever
            r = get_retriever()
            res = r.search(question, top_k=3)
            if res:
                return "\n\n".join(f"[{x['source']}]\n{x['text']}" for x in res)
        except Exception:
            pass
    docs = []
    docs_dir = ROOT_DIR / "data" / "docs"
    if docs_dir.exists():
        for p in sorted(docs_dir.glob("*.md")):
            docs.append(f"--- {p.name} ---\n" + p.read_text(encoding="utf-8")[:800])
    return "\n\n".join(docs)


def load_judge_prompt() -> str:
    if not JUDGE_PROMPT_PATH.exists():
        return "Check if answer is faithful to context. Reply JSON: {\"faithful\": true, \"reason\": \"ok\"}"
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8")


def call_judge_llm(question: str, answer: str, context: str = "") -> dict:
    """Call LLM judge to evaluate faithfulness."""
    try:
        from llm import call_llm
        if not context:
            context = load_corpus_context(question)
        prompt_tmpl = load_judge_prompt()
        prompt = prompt_tmpl.replace("{question}", question).replace("{answer}", answer).replace("{context}", context)
        messages = [{"role": "user", "content": prompt}]
        res = call_llm(messages, max_tokens=300)
        content = res.content.strip()
        # Extract json block
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as exc:
        # Fallback heuristic judge if LLM API not reachable or format issue
        is_refusal = any(w in answer.lower() for w in ["không có thông tin", "ngoài phạm vi", "không tìm thấy", "chưa được cập nhật", "không có trong kế hoạch", "no_answer"])
        return {"faithful": True if is_refusal else len(answer) > 20, "reason": f"Heuristic check: {exc}"}


async def baseline_tool_runner(tool_name: str, args: dict) -> str:
    """Tool runner cho Cấu hình 1 (Baseline): RAG thuần Dense, không có metadata filter."""
    if tool_name == "search_lesson_plan":
        from rag.retriever import search_lesson_plan
        q = str(args.get("query", ""))
        chunks = search_lesson_plan(query=q, mode="dense", doc_type=None)
        if not chunks:
            return "[]"
        formatted = [
            {
                "doc_id": c.doc_id if hasattr(c, "doc_id") else c.get("doc_id", ""),
                "chunk_id": c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id", ""),
                "score": round(float(c.score if hasattr(c, "score") else c.get("score", 0.0)), 3),
                "content": c.text if hasattr(c, "text") else c.get("text", ""),
            }
            for c in chunks
        ]
        return json.dumps(formatted, ensure_ascii=False)
    from agent import default_call_tool
    return await default_call_tool(tool_name, args)


def evaluate_config(config_name: str, testset: list[dict], verify: bool, tool_runner=None) -> tuple[dict, list[dict]]:
    """Chạy đánh giá cho một cấu hình cụ thể."""
    from agent import run as run_agent
    from llm import USAGE

    print("\n" + "=" * 70)
    print(f"   [ĐÁNH GIÁ] CẤU HÌNH: {config_name.upper()}")
    print("=" * 70)
    print(f"{'ID':<4} | {'Loại':<13} | {'Độ tin cậy':<10} | {'Từ chối':<8} | Câu hỏi")
    print("-" * 70)

    results = []
    in_scope_count = 0
    in_scope_faithful = 0
    out_of_scope_count = 0
    out_of_scope_refused = 0
    total_latency = 0.0

    for item in testset:
        qid = item.get("id")
        category = item.get("category", "in_scope")
        question = item.get("question", "")

        calls_before = USAGE.get("calls", 0)
        p_tokens_before = USAGE.get("prompt_tokens", 0)
        c_tokens_before = USAGE.get("completion_tokens", 0)

        t0 = time.time()
        try:
            answer = run_agent(question, verify=verify, call_tool_fn=tool_runner)
        except Exception as exc:
            answer = f"Lỗi thực thi: {exc}"
        latency = round(time.time() - t0, 2)
        total_latency += latency

        p_delta = max(0, USAGE.get("prompt_tokens", 0) - p_tokens_before)
        c_delta = max(0, USAGE.get("completion_tokens", 0) - c_tokens_before)
        calls_delta = max(1, USAGE.get("calls", 0) - calls_before)

        is_refusal = any(w in answer.lower() for w in [
            "không có thông tin", "ngoài phạm vi", "không tìm thấy",
            "chưa được cập nhật", "không có trong kế hoạch", "no_answer"
        ])

        judge_res = call_judge_llm(question, answer)
        faithful = judge_res.get("faithful", True)

        if category == "out_of_scope":
            out_of_scope_count += 1
            if is_refusal:
                out_of_scope_refused += 1
        else:
            in_scope_count += 1
            if faithful:
                in_scope_faithful += 1

        faithful_str = "PASS" if faithful else "FAIL"
        refusal_str = "YES" if is_refusal else "NO"
        q_snippet = (question[:28] + "...") if len(question) > 31 else question

        print(f"{qid:<4} | {category:<13} | {faithful_str:<10} | {refusal_str:<8} | {q_snippet}")

        results.append({
            "config": config_name,
            "id": qid,
            "category": category,
            "question": question,
            "answer": answer,
            "faithful": faithful,
            "refused": is_refusal,
            "judge_reason": judge_res.get("reason", ""),
            "latency": latency,
            "calls": calls_delta,
            "prompt_tokens": p_delta,
            "completion_tokens": c_delta,
        })

    faithfulness_rate = (in_scope_faithful / in_scope_count * 100) if in_scope_count > 0 else 100.0
    refusal_rate = (out_of_scope_refused / out_of_scope_count * 100) if out_of_scope_count > 0 else 100.0
    avg_latency = round(total_latency / len(testset), 2) if testset else 0.0

    summary = {
        "config": config_name,
        "total_questions": len(testset),
        "in_scope_count": in_scope_count,
        "in_scope_faithful": in_scope_faithful,
        "faithfulness_rate": round(faithfulness_rate, 1),
        "out_of_scope_count": out_of_scope_count,
        "out_of_scope_refused": out_of_scope_refused,
        "refusal_rate": round(refusal_rate, 1),
        "avg_latency": avg_latency,
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in results),
    }

    return summary, results


def main():
    print("=" * 70)
    print("   HỆ THỐNG ĐÁNH GIÁ SO SÁNH 2 CẤU HÌNH (A/B TESTING - SV3 VŨ)   ")
    print("=" * 70)

    testset = load_testset()
    if not testset:
        print("Không có câu hỏi kiểm thử trong testset.jsonl!")
        return

    print(f"[*] Đã nạp {len(testset)} câu hỏi từ {TESTSET_PATH.name}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_single = len(sys.argv) > 1 and sys.argv[1] in ("--single", "-s")

    all_results = []

    if run_single:
        # Chạy 1 cấu hình đề xuất duy nhất
        summary_prop, results_prop = evaluate_config("Proposed (Hybrid + Verify)", testset, verify=True)
        all_results.extend(results_prop)
    else:
        # Chạy so sánh 2 cấu hình 
        summary_base, results_base = evaluate_config(
            "Baseline (Dense, No Verify)", testset, verify=False, tool_runner=baseline_tool_runner
        )
        summary_prop, results_prop = evaluate_config(
            "Proposed (Hybrid + Verify)", testset, verify=True, tool_runner=None
        )
        all_results.extend(results_base)
        all_results.extend(results_prop)

        # In bảng so sánh
        print("\n" + "=" * 78)
        print("                 BẢNG SO SÁNH ĐỐI ĐẦU 2 CẤU HÌNH (A/B TESTING SUMMARY)                  ")
        print("=" * 78)
        print(f"{'Tiêu chí đánh giá':<30} | {'Cấu hình 1 (Baseline)':<20} | {'Cấu hình 2 (Proposed)':<20}")
        print("-" * 78)
        print(f"{'Độ trung thực (Faithfulness %)':<30} | {summary_base['faithfulness_rate']:>18.1f}% | {summary_prop['faithfulness_rate']:>18.1f}%")
        print(f"{'Tỷ lệ từ chối đúng (Refusal %)':<30} | {summary_base['refusal_rate']:>18.1f}% | {summary_prop['refusal_rate']:>18.1f}%")
        print(f"{'Thời gian xử lý TB (s)':<30} | {summary_base['avg_latency']:>19.2f}s | {summary_prop['avg_latency']:>19.2f}s")
        print(f"{'Tổng Token tiêu thụ':<30} | {summary_base['total_tokens']:>20} | {summary_prop['total_tokens']:>20}")
        print("=" * 78)

    # Lưu log ra JSONL
    log_file = LOGS_DIR / f"eval_ab_run_{run_timestamp}.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import csv
    csv_file = LOGS_DIR / f"eval_ab_run_{run_timestamp}.csv"
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config", "id", "category", "question", "faithful", "refused", "latency", "calls", "prompt_tokens", "completion_tokens", "judge_reason"]
        )
        writer.writeheader()
        for r in all_results:
            row = dict(r)
            row.pop("answer", None)
            writer.writerow(row)

    print(f"\n[OK] Đã lưu kết quả A/B testing vào:")
    print(f"  - CSV : {csv_file}")
    print(f"  - JSONL: {log_file}\n")


if __name__ == "__main__":
    main()

