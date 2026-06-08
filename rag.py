"""Retrieval + Grounded Generation - Stages 4, 5 of the pipeline.

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

TOP_K = 3
# Chroma cosine distance; chunks less similar than this are treated as irrelevant
# and dropped, so off-topic queries retrieve little/nothing and the model declines.
MAX_DISTANCE = 0.55
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are "The Unofficial Guide" - an assistant that answers \
questions for UC San Diego data science students using UCSD course documents, \
student planning notes, and r/UCSD discussion notes.

You must answer ONLY using the numbered CONTEXT passages provided in the user \
message. These passages are UCSD DSC notes retrieved for this question.

Rules:
- Base every claim on the CONTEXT. Do not use outside knowledge or invent facts, \
course numbers, prerequisites, policies, instructors, or requirements that are not in \
the CONTEXT.
- If the CONTEXT does not contain enough information to answer, say plainly: \
"I don't have enough information on that in my sources." Do not guess.
- Be specific: give the actual UCSD DSC details in the passages (e.g., "DSC 30", \
"DSC 80", "EASy request", "DSC 140A", "R/dataframe/notebook/project work").
- Attribute carefully: if a detail is about a specific course, prerequisite, or \
enrollment process, name it and do not mix up DSC, CSE, math, or GE advice.
- Cite the source file(s) you used at the end of your answer, like: \
"Sources: 05_dsc80_practice_application.txt".
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
        return "I don't have enough information on that in my sources."

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


def _replace_source_line(response: str, sources: list[str]) -> str:
    """Make source attribution deterministic instead of trusting LLM formatting."""
    lines = response.strip().splitlines()
    kept = []
    for line in lines:
        if line.strip().lower().startswith("sources:"):
            break
        kept.append(line)

    body = "\n".join(kept).rstrip()
    if not sources or "don't have enough information" in body.lower():
        return body

    return f"{body}\n\nSources: {', '.join(sources)}"


def answer(query: str, k: int = TOP_K) -> dict:
    """Full pipeline: retrieve then generate. Returns answer text + retrieved chunks."""
    chunks = retrieve(query, k=k)
    response = generate(query, chunks)
    sources = sorted({c["source"] for c in chunks})
    response = _replace_source_line(response, sources)
    return {"answer": response, "chunks": chunks, "sources": sources}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How should I plan the UCSD DSC lower-division sequence?"
    result = answer(q)
    print(f"Q: {q}\n")
    print(result["answer"])
    print("\nRetrieved from:", ", ".join(result["sources"]))
