"""Evaluation harness - runs the 5 planning.md test questions through the system.

    python evaluate.py

Prints, for each question: the retrieved source files (retrieval quality signal)
and the system's grounded answer. Use the output to fill the README Evaluation
Report and Failure Case Analysis sections. Includes a 6th, deliberately
out-of-corpus question to check that grounding makes the system decline.
"""

from rag import answer, retrieve

TEST_QUESTIONS = [
    "How should I plan the UCSD DSC lower-division sequence?",
    "Is DSC 30 a heavy class, and how should I handle it?",
    "What should I know before taking DSC 80?",
    "Can I assume an EASy request will let me take DSC 100 while still finishing a prerequisite?",
    "How should I prepare for DSC 140A?",
]

OUT_OF_CORPUS = "Which UCSD dining hall has the best late-night food?"


def run():
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print("=" * 80)
        print(f"Q{i}: {q}")
        hits = retrieve(q)
        srcs = ", ".join(f"{h['source']} (d={h['distance']:.3f})" for h in hits) or "(none)"
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
