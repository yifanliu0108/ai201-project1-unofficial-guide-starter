"""Retrieval + Grounded Generation — Stages 4, 5 of the pipeline.

`retrieve()` embeds the user query and pulls the top-k most similar chunks from
ChromaDB. `generate()` feeds those chunks to a Groq-hosted LLM under a strict
grounding system prompt that forbids answering beyond the retrieved context.
`answer()` ties them together and is what the interface (app.py) and evaluation
harness (evaluate.py) call.
"""

import os

import chromadb
from dotenv import load_dotenv
from groq import Groq

from ingest import CHROMA_DIR, COLLECTION_NAME, get_embedding_function

load_dotenv()

TOP_K = 4
# Chroma cosine distance; chunks less similar than this are treated as irrelevant
# and dropped, so off-topic queries retrieve little/nothing and the model declines.
MAX_DISTANCE = 0.9
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are "The Unofficial Guide" — a home-espresso assistant for \
beginners using entry-level machines (Breville Bambino, Gaggia Classic, etc.).

You must answer ONLY using the numbered CONTEXT passages provided in the user \
message. These passages are community knowledge retrieved for this question.

Rules:
- Base every claim on the CONTEXT. Do not use outside knowledge or invent facts, \
numbers, model names, or steps that are not in the CONTEXT.
- If the CONTEXT does not contain enough information to answer, say plainly: \
"I don't have information about that in my sources." Do not guess.
- Be specific and practical: give the actual numbers, steps, and reasoning from \
the passages (e.g., grind finer, 18 g in / 36 g out, wait for the light).
- Cite the source file(s) you used at the end of your answer, like: \
"Sources: 02_grind_taste_troubleshooting.md".
- Keep answers concise and directly responsive to the question."""


_collection = None


def _get_collection():
    """Lazily open the persistent Chroma collection built by ingest.py."""
    global _collection
    if _collection is None:
        if not os.path.isdir(CHROMA_DIR):
            raise SystemExit(
                f"No index found at '{CHROMA_DIR}/'. Run `python ingest.py` first."
            )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=get_embedding_function()
        )
    return _collection


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k chunks for ``query`` as dicts with text/source/distance.

    Chunks farther than MAX_DISTANCE are filtered out so that off-topic questions
    surface no (or weak) context and the generator declines instead of hallucinating.
    """
    collection = _get_collection()
    res = collection.query(query_texts=[query], n_results=k)

    hits: list[dict] = []
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for text, meta, dist in zip(docs, metas, dists):
        if dist <= MAX_DISTANCE:
            hits.append({"text": text, "source": meta["source"], "distance": dist})
    return hits


def _build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(blocks)


def generate(query: str, chunks: list[dict]) -> str:
    """Call the Groq LLM with the grounded system prompt and retrieved context."""
    if not chunks:
        return "I don't have information about that in my sources."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise SystemExit(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your free "
            "key from https://console.groq.com"
        )

    client = Groq(api_key=api_key)
    context = _build_context(chunks)
    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the CONTEXT above, and cite the source file(s)."
    )

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


def answer(query: str, k: int = TOP_K) -> dict:
    """Full pipeline: retrieve then generate. Returns answer text + retrieved chunks."""
    chunks = retrieve(query, k=k)
    response = generate(query, chunks)
    sources = sorted({c["source"] for c in chunks})
    return {"answer": response, "chunks": chunks, "sources": sources}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "My shot pulls fast and tastes sour. What do I do?"
    result = answer(q)
    print(f"Q: {q}\n")
    print(result["answer"])
    print("\nRetrieved from:", ", ".join(result["sources"]))
