"""Query interface — Milestone 5.

A Gradio chat UI over the RAG pipeline. Run with:

    python app.py

Then open the printed local URL. Ask espresso questions; the assistant answers
only from the indexed community documents and cites its sources.
"""

import gradio as gr

from rag import answer


EXAMPLES = [
    "My espresso shot pulls really fast and tastes sour. What should I change?",
    "What dose and ratio should I use for a double shot on a Breville Bambino?",
    "How do I fix channeling in my espresso puck?",
    "What brew temperature should I target on a Gaggia Classic without a PID?",
]


def respond(message: str, history):
    """Gradio ChatInterface callback: return the grounded answer with sources."""
    if not message.strip():
        return "Ask me a home-espresso question and I'll answer from my sources."
    result = answer(message)
    text = result["answer"]
    if result["sources"] and "don't have information" not in text:
        # Show which documents were retrieved, for transparency.
        retrieved = ", ".join(result["sources"])
        text += f"\n\n_Retrieved from: {retrieved}_"
    return text


demo = gr.ChatInterface(
    fn=respond,
    title="☕ The Unofficial Guide — Home Espresso",
    description=(
        "A retrieval-augmented assistant grounded in community espresso knowledge "
        "(r/espresso, Home-Barista). It answers only from its indexed documents and "
        "cites them. Run `python ingest.py` first to build the index."
    ),
    examples=EXAMPLES,
)


if __name__ == "__main__":
    demo.launch()
