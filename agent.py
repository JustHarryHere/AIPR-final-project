import asyncio
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from llm import USAGE, call_llm
from schemas import Action, parse_action

MAX_STEPS = 8
WRITE_TOOLS = {"create_study_plan_item"}
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system.txt"
LOGS_DIR = Path(__file__).resolve().parent / "logs"

#fake memory for testing
_study_plan_db: dict[str, list[dict]] = {}


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Bạn là trợ lý học tập. Trả về JSON action."


def log_trace(question: str, steps: list[dict], answer: str) -> None:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "agent_trace.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "question": question,
                "steps": steps,
                "answer": answer,
                "usage": dict(USAGE),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


async def default_call_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Fallback tool runner: connects to MCP server if present, else uses mock."""
    server_file = Path(__file__).resolve().parent / "mcp_server" / "server.py"
    if server_file.is_file():
        try:
            from mcp import Client, StdioServerParameters
            server = StdioServerParameters(
                command=sys.executable,
                args=["-u", str(server_file)],
                cwd=str(server_file.parent.parent),
            )
            async with Client(server) as client:
                res = await client.call_tool(tool_name, arguments=args)
                content = getattr(res, "content", None)
                if content:
                    return "\n".join(getattr(c, "text", str(c)) for c in content)
                return str(res)
        except Exception:
            pass

    try:
        if tool_name == "calc_required_score":
            from rag.calc import calc_required_score
            res = calc_required_score(
                midterm=float(args.get("midterm", 0)),
                w_mid=float(args.get("w_mid", 0.4)),
                w_final=float(args.get("w_final", 0.6)),
                target=float(args.get("target", 8.0)),
            )
            return json.dumps(res.get("data", res), ensure_ascii=False)

        elif tool_name == "calc_workload":
            from rag.calc import calc_workload
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
            chunks = search_lesson_plan(query=q, mode=mode)
            if not chunks:
                return "[]"
            # Format chunks nicely for agent prompt
            formatted = [
                {
                    "doc_id": c["doc_id"],
                    "chunk_id": c["chunk_id"],
                    "score": round(c["score"], 3),
                    "content": c["text"],
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


async def run_agent(
    question: str,
    call_tool_fn: Optional[Callable] = None,
    verbose: bool = False,
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
            log_trace(question, steps_log, answer)
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
            print(f"[{step + 1}] {action.tool}({action.args}) → {result}")

        # verify tool
        if action.tool in WRITE_TOOLS:
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
                print(f"[{step + 1}.verify] list_study_plan({{'student_id': '{student_id}'}}) → {check}  ← VERIFY")

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"{action.tool} → {result}\nVERIFY list_study_plan → {check}"})
        else:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"{action.tool} → {result}"})

    fallback_ans = "Không hoàn thành trong số bước cho phép."
    log_trace(question, steps_log, fallback_ans)
    return fallback_ans


def run(question: str, verbose: bool = False) -> str:
    """Hàm đồng bộ để gọi agent từ CLI."""
    return asyncio.run(run_agent(question, verbose=verbose))
