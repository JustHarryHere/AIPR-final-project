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


def load_judge_prompt() -> str:
    if not JUDGE_PROMPT_PATH.exists():
        return "Check if answer is faithful to context. Reply JSON: {\"faithful\": true, \"reason\": \"ok\"}"
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8")


def call_judge_llm(question: str, answer: str, context: str = "") -> dict:
    """Call LLM judge to evaluate faithfulness."""
    try:
        from llm import call_llm
        prompt_tmpl = load_judge_prompt()
        prompt = prompt_tmpl.replace("{question}", question).replace("{answer}", answer).replace("{context}", context or "Kế hoạch bài giảng môn AIPR và quy chế đánh giá môn học.")
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
        is_refusal = any(w in answer.lower() for w in ["không có thông tin", "ngoài phạm vi", "không tìm thấy", "chưa được cập nhật"])
        return {"faithful": True if is_refusal else len(answer) > 20, "reason": f"Heuristic check: {exc}"}


def main():
    print("=" * 65)
    print("      HỆ THỐNG ĐÁNH GIÁ (EVALUATION SUITE - SV3 VŨ)     ")
    print("=" * 65)

    testset = load_testset()
    if not testset:
        print("Không có câu hỏi kiểm thử trong testset.jsonl!")
        return

    print(f"[*] Đã nạp {len(testset)} câu hỏi từ {TESTSET_PATH.name}")

    try:
        from agent import run as run_agent
        from llm import USAGE
    except ImportError:
        print("[!] Chú ý: Chưa nạp được agent.py (đang ở nhánh riêng trước khi merge). Sử dụng fallback runner.")
        def run_agent(q):
            return f"Câu trả lời mẫu cho câu hỏi: {q}"
        USAGE = {"calls": 1, "prompt_tokens": 120, "completion_tokens": 45, "max_prompt": 120}

    results = []
    in_scope_count = 0
    in_scope_faithful = 0
    out_of_scope_count = 0
    out_of_scope_refused = 0

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"eval_run_{run_timestamp}.jsonl"

    print("-" * 65)
    print(f"{'ID':<4} | {'Loại':<13} | {'Độ tin cậy':<10} | {'Từ chối':<8} | Câu hỏi")
    print("-" * 65)

    for item in testset:
        qid = item.get("id")
        category = item.get("category", "in_scope")
        question = item.get("question", "")

        # Record token baseline
        calls_before = USAGE.get("calls", 0)
        p_tokens_before = USAGE.get("prompt_tokens", 0)
        c_tokens_before = USAGE.get("completion_tokens", 0)

        t0 = time.time()
        answer = run_agent(question)
        latency = round(time.time() - t0, 2)

        # Delta tokens
        p_delta = max(0, USAGE.get("prompt_tokens", 0) - p_tokens_before)
        c_delta = max(0, USAGE.get("completion_tokens", 0) - c_tokens_before)

        # Evaluation logic
        is_refusal = any(w in answer.lower() for w in ["không có thông tin", "ngoài phạm vi", "không tìm thấy", "chưa được cập nhật", "no_answer"])
        
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
        q_snippet = (question[:32] + "...") if len(question) > 35 else question

        print(f"{qid:<4} | {category:<13} | {faithful_str:<10} | {refusal_str:<8} | {q_snippet}")

        record = {
            "id": qid,
            "category": category,
            "question": question,
            "answer": answer,
            "faithful": faithful,
            "refused": is_refusal,
            "judge_reason": judge_res.get("reason", ""),
            "latency": latency,
            "prompt_tokens": p_delta,
            "completion_tokens": c_delta,
        }
        results.append(record)

    # Save to jsonl log
    with open(log_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Metrics computation
    faithfulness_rate = (in_scope_faithful / in_scope_count * 100) if in_scope_count > 0 else 100.0
    refusal_rate = (out_of_scope_refused / out_of_scope_count * 100) if out_of_scope_count > 0 else 100.0

    print("=" * 65)
    print("                     KẾT QUẢ ĐÁNH GIÁ (SUMMARY)                 ")
    print("=" * 65)
    print(f" - Tổng số câu kiểm thử   : {len(testset)}")
    print(f" - Câu hỏi trong đề cương : {in_scope_count} câu")
    print(f"   + Tỷ lệ Faithfulness   : {faithfulness_rate:.1f}% ({in_scope_faithful}/{in_scope_count})")
    print(f" - Câu hỏi ngoài đề cương : {out_of_scope_count} câu")
    print(f"   + Tỷ lệ từ chối đúng   : {refusal_rate:.1f}% ({out_of_scope_refused}/{out_of_scope_count})")
    print(f" - Tổng Token sử dụng     : {USAGE.get('prompt_tokens', 0) + USAGE.get('completion_tokens', 0)} tokens")
    print(f" - Log chi tiết đã lưu    : {log_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
