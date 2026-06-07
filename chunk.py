"""Chunking — Stage 2 of the pipeline.

Splits document text into ~600-character chunks with 100 characters of overlap,
preferring sentence/paragraph boundaries so a single tip or procedure step is not
cut in half. See planning.md "Chunking Strategy" for the rationale.
"""

import re

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-ish units, keeping paragraph breaks as boundaries."""
    # Split on blank lines (paragraphs) first, then on sentence terminators.
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Split after . ! ? : or newline, keeping the delimiter with the sentence.
        for sent in re.split(r"(?<=[.!?:])\s+|\n", paragraph):
            sent = sent.strip()
            if sent:
                pieces.append(sent)
    return pieces


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split ``text`` into chunks of about ``size`` characters with ``overlap``.

    Chunks are assembled from whole sentences so we never cut mid-word; when a
    chunk is emitted, the next chunk starts ``overlap`` characters back into the
    previous chunk's tail so context that straddles a boundary is preserved.
    """
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        # A single sentence longer than the chunk size: hard-wrap it on whitespace.
        if len(sent) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            for hard in _hard_wrap(sent, size, overlap):
                chunks.append(hard.strip())
            continue

        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current.strip())
            # Start the next chunk with the overlap tail of the one we just closed.
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sent}".strip() if tail else sent

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def _hard_wrap(sentence: str, size: int, overlap: int) -> list[str]:
    """Wrap an over-long sentence on whitespace into overlapping windows."""
    words = sentence.split()
    out: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= size:
            current = candidate
        else:
            out.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {word}".strip() if tail else word
    if current:
        out.append(current)
    return out


if __name__ == "__main__":
    sample = "First sentence here. Second sentence here. " * 40
    cs = chunk_text(sample)
    print(f"{len(cs)} chunks; sizes: {[len(c) for c in cs]}")
