import os
import sys

from agent import MAX_STEPS, run, get_trace_log_file
from llm import USAGE

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_DEMO_QUESTION = (
    "Lập kế hoạch học tập môn AIPR tuần 4 chủ đề Xây dựng RAG Pipeline cơ bản "
    "với 4 giờ tự học cho sinh viên sv01."
)
WIDTH = 58


def line(char="=", width=WIDTH):
    print(char * width)


def print_banner():
    line("=")
    print("       AI-Powered Study Assistant ")
    line("=")
    print("[INIT] Dang khoi tao he thong...")
    print("[ OK ] ReAct Agent san sang.")
    print(f"[INFO] Gioi han agent loop: {MAX_STEPS} steps")
    line("=")


def print_usage_summary():
    prompt_tokens = int(USAGE.get("prompt_tokens", 0))
    completion_tokens = int(USAGE.get("completion_tokens", 0))
    total_tokens = prompt_tokens + completion_tokens

    line("=")
    print("TOKEN USAGE & COST")
    line("-")
    print(f"LLM calls          : {USAGE.get('calls', 0)}")
    print(f"Retry attempts     : {USAGE.get('retries', 0)}")
    print(f"Input tokens       : {prompt_tokens}")
    print(f"Output tokens      : {completion_tokens}")
    print(f"Total tokens       : {total_tokens}")
    print(f"Max prompt         : {USAGE.get('max_prompt', 0)}")

    # Only estimate cost when the user configures the model's rates.
    input_rate = os.getenv("INPUT_COST_PER_MILLION")
    output_rate = os.getenv("OUTPUT_COST_PER_MILLION")
    if input_rate is not None and output_rate is not None:
        try:
            cost = (prompt_tokens * float(input_rate) + completion_tokens * float(output_rate)) / 1_000_000
            print(f"Estimated cost     : ${cost:.8f} USD")
        except ValueError:
            print("Estimated cost     : unavailable (invalid cost rates)")
    else:
        print("Estimated cost     : not configured")

    print(f"Trace log          : {get_trace_log_file()}")
    print(f"Cumulative trace   : {os.path.join('logs', 'agent_trace.jsonl')}")
    line("=")


def main():
    print_banner()
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--demo", "-d"):
            question = DEFAULT_DEMO_QUESTION
        else:
            question = " ".join(sys.argv[1:])
    else:
        try:
            question = input("[INPUT] Cau hoi: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Da huy.")
            return

    if not question:
        question = DEFAULT_DEMO_QUESTION

    line("-")
    print("[QUESTION]")
    print(question)
    line("-")
    print("[PROCESS] Bat dau ReAct Agent loop...\n")

    try:
        answer = run(question, verbose=True)
        print("\n[FINAL ANSWER]")
        line("-")
        print(answer)
        line("-")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Da dung theo yeu cau.")
    except Exception as exc:
        print(f"\n[ERROR] Agent that bai: {type(exc).__name__}: {exc}")
    finally:
        print_usage_summary()


if __name__ == "__main__":
    main()
