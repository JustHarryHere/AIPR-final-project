import sys
from agent import run

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        try:
            question = input("Câu hỏi: ").strip()
        except (KeyboardInterrupt, EOFError):
            return

    if not question:
        return

    print(f"\nCâu hỏi:\n{question}\n")
    print("Quá trình xử lý:")
    answer = run(question, verbose=True)
    print(f"\nTrả lời:\n{answer}\n")


if __name__ == "__main__":
    main()
