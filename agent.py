import asyncio
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from llm import USAGE, call_llm, reset_usage
from schemas import Action, parse_action

MAX_STEPS = 8
WRITE_TOOLS = {"create_study_plan_item"}
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system.txt"
LOGS_DIR = Path(__file__).resolve().parent / "logs"

_session_timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_trace_log_file() -> Path:
    global _session_timestamp
    return LOGS_DIR / f"agent_trace_{_session_timestamp}.jsonl"

#fake memory for testing
_study_plan_db: dict[str, list[dict]] = {}


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Bạn là trợ lý học tập. Trả về JSON action."


def get_question_id(question: str) -> Any:
    testset_file = Path(__file__).resolve().parent / "tests" / "testset.jsonl"
    if testset_file.exists():
        try:
            q_clean = question.strip().lower()
            with open(testset_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        iq = item.get("question", "").strip().lower()
                        if iq == q_clean or iq in q_clean or q_clean in iq:
                            return item.get("id")
        except Exception:
            pass
    log_file = LOGS_DIR / "agent_trace.jsonl"
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return len([l for l in f if l.strip()]) + 1
        except Exception:
            pass
    return 1


def log_trace(question: str, steps: list[dict], answer: str, qid: Any = None) -> None:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if qid is None:
            qid = get_question_id(question)
        record = {
            "id": qid,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "steps": steps,
            "answer": answer,
            "usage": dict(USAGE),
        }
        # 1. Ghi vào file trace riêng của từng phiên chạy có timestamp (chuẩn rubric: >= 3 file chạy thật có timestamp)
        session_file = get_trace_log_file()
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2. Đồng thời ghi vào agent_trace.jsonl để lưu lịch sử tích lũy
        main_file = LOGS_DIR / "agent_trace.jsonl"
        with open(main_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass




async def default_call_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Tool runner: connects to MCP server via stdio, else uses local fallback."""
    server_file = Path(__file__).resolve().parent / "mcp_server" / "server.py"
    if server_file.is_file():
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            server = StdioServerParameters(
                command=sys.executable,
                args=["-u", str(server_file)],
                cwd=str(server_file.parent.parent),
            )
            async with stdio_client(server) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    res = await client.call_tool(tool_name, arguments=args)
                    content = getattr(res, "content", None)
                    if content:
                        return "\n".join(getattr(c, "text", str(c)) for c in content)
                    return str(res)
        except Exception:
            pass

    try:
        if tool_name == "calc_required_score":
            from mcp_server.server import calc_required_score
            res = calc_required_score(
                midterm=float(args.get("midterm", 0)),
                w_mid=float(args.get("w_mid", 0.4)),
                w_final=float(args.get("w_final", 0.6)),
                target=float(args.get("target", 8.0)),
            )
            return json.dumps(res.get("data", res), ensure_ascii=False)

        elif tool_name == "calc_workload":
            from mcp_server.server import calc_workload
            res = calc_workload(
                hours_per_week=float(args.get("hours_per_week", 4)),
                from_week=int(args.get("from_week", 1)),
                to_week=int(args.get("to_week", 5)),
            )
            return json.dumps(res.get("data", res), ensure_ascii=False)

        elif tool_name == "search_lesson_plan":
            from rag.retriever import search_lesson_plan
            q = str(args.get("query", ""))
            mode = str(args.get("mode", "hybrid"))
            doc_type = args.get("doc_type")
            chunks = search_lesson_plan(query=q, mode=mode, doc_type=doc_type)
            if not chunks:
                return "[]"
            # Format chunks nicely for agent prompt
            formatted = [
                {
                    "doc_id": c["doc_id"] if isinstance(c, dict) else getattr(c, "doc_id", ""),
                    "chunk_id": c["chunk_id"] if isinstance(c, dict) else getattr(c, "chunk_id", ""),
                    "score": round(c["score"] if isinstance(c, dict) else getattr(c, "score", 0.0), 3),
                    "content": c["text"] if isinstance(c, dict) else getattr(c, "text", getattr(c, "content", "")),
                }
                for c in chunks
            ]
            return json.dumps(formatted, ensure_ascii=False)

        elif tool_name == "get_course_info":
            cat_file = Path(__file__).resolve().parent / "data" / "course_catalog.json"
            cid = str(args.get("course_id", "AIPR")).upper()
            if cat_file.exists():
                cat = json.loads(cat_file.read_text(encoding="utf-8"))
                if cid in cat:
                    return json.dumps(cat[cid], ensure_ascii=False)
            return json.dumps({"course_id": cid, "w_mid": 0.4, "w_final": 0.6}, ensure_ascii=False)

        elif tool_name == "create_study_plan_item":
            plan_file = Path(__file__).resolve().parent / "data" / "study_plan.json"
            plan_file.parent.mkdir(parents=True, exist_ok=True)
            plans = []
            if plan_file.exists():
                try:
                    plans = json.loads(plan_file.read_text(encoding="utf-8"))
                except Exception:
                    plans = []
            import uuid
            new_item = {
                "id": str(uuid.uuid4())[:8],
                "student_id": str(args.get("student_id", "sv01")),
                "course": str(args.get("course", "AIPR")),
                "week": int(args.get("week", 1)),
                "topic": str(args.get("topic", "")),
                "hours": float(args.get("hours", 4.0)),
            }
            plans.append(new_item)
            plan_file.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"OK: added item id={new_item['id']} for {new_item['student_id']}"

        elif tool_name == "list_study_plan":
            plan_file = Path(__file__).resolve().parent / "data" / "study_plan.json"
            sid = str(args.get("student_id", "sv01"))
            if not plan_file.exists():
                return "[]"
            try:
                plans = json.loads(plan_file.read_text(encoding="utf-8"))
                user_plans = [p for p in plans if p.get("student_id") == sid]
                return json.dumps(user_plans, ensure_ascii=False)
            except Exception:
                return "[]"

        return f"Error: unknown tool {tool_name}"
    except Exception as exc:
        return f"Error: {exc}"


def _print_tool_result(tool_name: str, result: str) -> None:
    """Print concise, readable tool output while preserving full output in trace/messages."""
    print(f"[TOOL RESULT] {tool_name}")
    try:
        parsed = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        text = str(result).strip()
        print(f"   {text[:500]}{'...' if len(text) > 500 else ''}")
        return

    if tool_name == "search_lesson_plan" and isinstance(parsed, list):
        print(f"   Retrieved chunks: {len(parsed)}")
        for i, item in enumerate(parsed[:5], 1):
            if not isinstance(item, dict):
                continue
            content = ' '.join(str(item.get("content", "")).split())
            if len(content) > 220:
                content = content[:217] + "..."
            print(f"   [{i}] doc={item.get('doc_id', '?')} | chunk={item.get('chunk_id', '?')} | score={item.get('score', '?')}")
            if content:
                print(f"       {content}")
        if len(parsed) > 5:
            print(f"   ... {len(parsed) - 5} result(s) omitted from terminal; full data is retained in trace.")
    elif tool_name == "list_study_plan" and isinstance(parsed, list):
        print(f"   Items found: {len(parsed)}")
        for item in parsed[-3:]:
            if isinstance(item, dict):
                print(f"   - week={item.get('week', '?')} | topic={item.get('topic', '?')} | hours={item.get('hours', '?')} | id={item.get('id', '?')}")
    elif isinstance(parsed, dict):
        compact = json.dumps(parsed, ensure_ascii=False)
        print(f"   {compact[:500]}{'...' if len(compact) > 500 else ''}")
    else:
        print(f"   {str(parsed)[:500]}")


async def run_agent(
    question: str,
    call_tool_fn: Optional[Callable] = None,
    verbose: bool = False,
    verify: bool = True,
    qid: Optional[Any] = None,
) -> str:
    """Vòng lặp ReAct agent tự viết theo chuẩn T3 Section A4."""
    call_tool = call_tool_fn or default_call_tool
    system_prompt = load_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Câu hỏi: {question}"},
    ]

    steps_log: list[dict[str, Any]] = []

    for step in range(MAX_STEPS):
        reply = call_llm(messages)
        content = reply.content or ""

        action = parse_action(content)
        if action is None:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Error: không đọc được action, hãy trả JSON đúng format"})
            steps_log.append({"step": step + 1, "raw": content, "error": "parse_action failed"})
            continue

        if action.type == "final":
            answer = action.answer or "Hoàn thành."
            steps_log.append({"step": step + 1, "type": "final", "answer": answer})
            log_trace(question, steps_log, answer, qid=qid)
            return answer

        # Execute tool
        try:
            if inspect.iscoroutinefunction(call_tool):
                result = await call_tool(action.tool, action.args)
            else:
                result = call_tool(action.tool, action.args)
        except Exception as exc:
            result = f"Error: {exc}"

        if not result or result == "[]" or result == "{}":
            result = "Không tìm thấy thông tin phù hợp trong kế hoạch học tập môn học."

        steps_log.append({
            "step": step + 1,
            "tool": action.tool,
            "args": action.args,
            "result": result,
        })

        if verbose:
            print(f"[STEP {step + 1}/{MAX_STEPS}] Tool: {action.tool}")
            print(f"[TOOL CALL] {json.dumps(action.args, ensure_ascii=False)}")
            _print_tool_result(action.tool, result)

        # verify tool
        if verify and action.tool in WRITE_TOOLS:
            student_id = action.args.get("student_id", "sv01")
            try:
                if inspect.iscoroutinefunction(call_tool):
                    check = await call_tool("list_study_plan", {"student_id": student_id})
                else:
                    check = call_tool("list_study_plan", {"student_id": student_id})
            except Exception as exc:
                check = f"Error: {exc}"

            steps_log.append({
                "step": f"{step + 1}.verify",
                "tool": "list_study_plan",
                "result": check,
            })

            if verbose:
                print(f"[VERIFY] list_study_plan(student_id={student_id!r})")
                try:
                    verified_items = json.loads(check)
                    verified = isinstance(verified_items, list) and any(
                        item.get("student_id") == student_id
                        and item.get("topic") == action.args.get("topic")
                        and item.get("week") == action.args.get("week")
                        for item in verified_items if isinstance(item, dict)
                    )
                except (TypeError, json.JSONDecodeError):
                    verified = False
                print("[VERIFY: PASS] Created item found in study plan." if verified else "[VERIFY: CHECK] Could not confirm exact item; inspect full trace.")
                _print_tool_result("list_study_plan", check)

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"{action.tool} → {result}\nVERIFY list_study_plan → {check}"})
        else:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"{action.tool} → {result}"})

    fallback_ans = "Không hoàn thành trong số bước cho phép."
    log_trace(question, steps_log, fallback_ans, qid=qid)
    return fallback_ans


def run(question: str, verbose: bool = False, verify: bool = True, call_tool_fn: Optional[Callable] = None, qid: Optional[Any] = None) -> str:
    """Hàm đồng bộ để gọi agent từ CLI."""
    reset_usage()
    return asyncio.run(run_agent(question, call_tool_fn=call_tool_fn, verbose=verbose, verify=verify, qid=qid))


