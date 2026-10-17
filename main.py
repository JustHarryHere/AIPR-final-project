import sys
from agent import run

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


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--demo", "-d"):
            question = DEFAULT_DEMO_QUESTION
        else:
            question = " ".join(sys.argv[1:])
    else:
        try:
            question = input("Câu hỏi: ").strip()
        except (KeyboardInterrupt, EOFError):
            return

    if not question:
        question = DEFAULT_DEMO_QUESTION

    print(f"\nCâu hỏi:\n{question}\n")
    print("Quá trình xử lý:")
    answer = run(question, verbose=True)
    print(f"\nTrả lời:\n{answer}\n")


if __name__ == "__main__":
    main()
