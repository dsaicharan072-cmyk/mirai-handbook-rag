"""Test runner for Mirai Academic & Hostel Policy Advisor RAG.
Executes assessment questions Q1 to Q5, prints outputs and retrieved chunks,
and saves results to outputs/results.json.
"""

import json
from pathlib import Path

from src.config import OUTPUTS_DIR, TOP_K, DISTANCE_THRESHOLD
from src.rag import answer, retrieve_context

TEST_QUESTIONS = [
    {
        "id": "Q1",
        "question": "I have 72% attendance this semester. How many attendance marks will I receive, and am I eligible to appear for the final examination?",
    },
    {
        "id": "Q2",
        "question": "Within how many days after recovery must a student submit supporting documents for Medical Leave? What happens if the documents are submitted after this period?",
    },
    {
        "id": "Q3",
        "question": "What is the monthly subscription fee for using the campus swimming pool and indoor squash court?",
    },
    {
        "id": "Q4",
        "question": "What minimum percentage of batch-student support is required to propose a new student club, and which authority reviews the proposal first?",
    },
    {
        "id": "Q5",
        "question": "What is the default WPA2 password for the Mirai_Student_5G campus Wi-Fi network?",
    },
]


def run_all_tests():
    print("=" * 80)
    print("MIRAI ACADEMIC & HOSTEL POLICY ADVISOR - RAG TEST SUITE")
    print("=" * 80)

    results = []

    for item in TEST_QUESTIONS:
        qid = item["id"]
        q_text = item["question"]

        print(f"\n{'='*30} [{qid}] {'='*30}")
        print(f"QUESTION: {q_text}\n")

        # Retrieve top chunks for inspection
        retrieved = retrieve_context(q_text, top_k=TOP_K, distance_threshold=DISTANCE_THRESHOLD)
        print("--- TOP RETRIEVED CHUNKS ---")
        if not retrieved:
            print("  (No chunks passed relevance threshold)")
        else:
            for idx, c in enumerate(retrieved, start=1):
                preview = c["text"][:140].replace("\n", " ")
                print(f"  [{idx}] Chunk ID: {c['chunk_id']} | Page: {c['page']} | Cosine Distance: {c['distance']:.4f}")
                print(f"      Text: \"{preview}...\"")

        # Run answer pipeline
        print("\n--- GENERATED ANSWER ---")
        output = answer(q_text)
        print(f"Answer:\n{output['answer']}\n")
        print(f"Sources: {output['sources']}")

        results.append(
            {
                "id": qid,
                "question": q_text,
                "answer": output["answer"],
                "sources": output["sources"],
                "retrieved_chunks": [
                    {
                        "chunk_id": c["chunk_id"],
                        "page": c["page"],
                        "distance": round(c["distance"], 4),
                    }
                    for c in retrieved
                ],
            }
        )

    # Save to outputs/results.json
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUTS_DIR / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"All tests completed. Output saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
