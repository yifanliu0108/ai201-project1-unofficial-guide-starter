"""Query interface - Milestone 5.

A Gradio chat UI over the RAG pipeline. Run with:

    python app.py

Then open the printed local URL. Ask UCSD data science course-planning questions; the
assistant answers only from the indexed community documents and cites its sources.
"""

import gradio as gr

from rag import answer


EXAMPLES = [
    "How should I plan the UCSD DSC lower-division sequence?",
    "Is DSC 30 a heavy class, and how should I handle it?",
    "What should I know before taking DSC 80?",
    "How should I prepare for DSC 140A?",
]


def respond(message: str, history):
    """Gradio ChatInterface callback: return the grounded answer with sources."""
    if not message.strip():
        return "Ask me about UCSD data science courses and I'll answer from the indexed sources."
    result = answer(message)
    text = result["answer"]
    if result["sources"] and "don't have enough information" not in text:
        # Show which documents were retrieved, for transparency.
        retrieved = ", ".join(result["sources"])
        text += f"\n\n_Retrieved from: {retrieved}_"
    return text


demo = gr.ChatInterface(
    fn=respond,
    title="The Unofficial Guide - UCSD Data Science",
    description=(
        "A retrieval-augmented assistant grounded in UCSD data science course pages, "
        "student planning notes, and r/UCSD discussion notes. It answers only from its "
        "indexed documents and cites them. Run `python ingest.py` first to build the "
        "index."
    ),
    examples=EXAMPLES,
)


if __name__ == "__main__":
    demo.launch()
