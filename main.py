# ============================================================
# main.py
# Entry point for the Sunglasses Store Code-Execution Agent.
# Run: python main.py
#
# Uses the same example questions as the M5_UGL_1_R.ipynb notebook.
# ============================================================

from dotenv import load_dotenv
load_dotenv()

from agent import run_agent


def divider(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# Run the same questions from the notebook
# ============================================================

if __name__ == "__main__":

    questions = [
        # Notebook Section 2.3 — Andrew's read-only example
        "Do you have any round sunglasses in stock that are under $100?",

        # Notebook Section 2.4 — return scenario (read-only: agent describes what would happen)
        "I want to return 2 Aviator sunglasses I bought last week.",

        # Notebook Section 4 — purchase multiple items (read-only: describe availability)
        "I want to buy 3 pairs of Classic sunglasses and 1 pair of Aviator sunglasses.",

        # Edge case: item that doesn't exist
        "Do you have any titanium frames?",

        # Edge case: out-of-stock scenario
        "I need 100 Mystique sunglasses for a corporate event.",
    ]

    for q in questions:
        divider(f"Question: {q}")
        try:
            answer = run_agent(q)
            print(f"\n  >>> ANSWER: {answer}\n")
        except Exception as e:
            print(f"\n  >>> ERROR: {e}\n")

    # ============================================================
    # Interactive mode — type your own questions
    # ============================================================
    divider("Interactive Mode — type 'quit' to exit")
    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        try:
            answer = run_agent(user_input)
            print(f"\n>>> {answer}")
        except Exception as e:
            print(f"Error: {e}")