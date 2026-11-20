"""Interactive Terminal Chatbot CLI for Mirai Academic & Hostel Policy Advisor."""

import sys
from src.rag import answer, retrieve_context
from src.config import TOP_K, DISTANCE_THRESHOLD

BANNER = r"""
========================================================================
       MIRAI ACADEMIC & HOSTEL POLICY ADVISOR - CHATBOT
========================================================================
Ask any question about university policies, attendance rules, medical leaves,
hostel curfew, student clubs, or campus guidelines.
Type 'exit', 'quit', or 'q' to end the session.
Type 'help' for sample questions.
========================================================================
"""

SAMPLE_QUESTIONS = [
    "I have 72% attendance this semester. How many attendance marks will I receive, and am I eligible to appear for the final examination?",
    "Within how many days after recovery must a student submit supporting documents for Medical Leave? What happens if the documents are submitted after this period?",
    "What minimum percentage of batch-student support is required to propose a new student club, and which authority reviews the proposal first?",
    "What is the late fee charged per day for outstanding fee dues?",
    "What is the dress code policy for hackathons and external competitions?",
    "What is the default WPA2 password for the Mirai_Student_5G campus Wi-Fi network?",
]


def start_chat():
    print(BANNER)

    while True:
        try:
            user_input = input("\nStudent: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chatbot. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Thank you for using Mirai Policy Advisor. Have a great day!")
            break

        if user_input.lower() == "help":
            print("\nSample questions you can ask:")
            for i, sq in enumerate(SAMPLE_QUESTIONS, 1):
                print(f"  {i}. {sq}")
            continue

        print("\nThinking and searching handbook...")

        # Run answer pipeline
        res = answer(user_input)
        ans_text = res["answer"]
        sources = res["sources"]

        print("\nAdvisor:")
        print(ans_text)

        if sources:
            print(f"\n[Cited Sources: {', '.join(str(s) for s in sources)}]")
        elif ans_text == "[not found]":
            print("\n(Note: The handbook does not contain information to answer this question.)")


if __name__ == "__main__":
    start_chat()
