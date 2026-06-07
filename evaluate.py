"""Evaluation harness — runs the 5 planning.md test questions through the system.

    python evaluate.py

Prints, for each question: the retrieved source files (retrieval quality signal)
and the system's grounded answer. Use the output to fill the README Evaluation
Report and Failure Case Analysis sections. Includes a 6th, deliberately
out-of-corpus question to check that grounding makes the system decline.
"""

from rag import answer, retrieve

TEST_QUESTIONS = [
    "My espresso shot pulls really fast and tastes sour. What should I change?",
    "What dose and ratio should I use for a double shot on a Breville Bambino?",
    "The steam on my older Gaggia Classic is weak and bubbly. How do I get better microfoam?",
    "My shots spray everywhere and taste both harsh and sour. How do I fix channeling?",
    "What brew water temperature should I target, and how do I hit it on a Gaggia Classic without a PID?",
]

OUT_OF_CORPUS = "What's the best grinder burr size for a Niche Zero, and how do I season it?"


def run():
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print("=" * 80)
        print(f"Q{i}: {q}")
        hits = retrieve(q)
        srcs = ", ".join(f"{h['source']} (d={h['distance']:.3f})" for h in hits)
        print(f"Retrieved: {srcs}")
        print("-" * 80)
        print(answer(q)["answer"])
        print()

    print("=" * 80)
    print("OUT-OF-CORPUS GROUNDING CHECK")
    print(f"Q: {OUT_OF_CORPUS}")
    hits = retrieve(OUT_OF_CORPUS)
    srcs = ", ".join(f"{h['source']} (d={h['distance']:.3f})" for h in hits) or "(none)"
    print(f"Retrieved: {srcs}")
    print("-" * 80)
    print(answer(OUT_OF_CORPUS)["answer"])


if __name__ == "__main__":
    run()
