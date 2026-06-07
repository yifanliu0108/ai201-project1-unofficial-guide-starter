# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

**Home espresso for entry-level machines — dialing in shots and machine technique.**

The domain is the practical, hands-on knowledge needed to pull good espresso on affordable home machines (Breville Bambino, Gaggia Classic Pro, and similar). This knowledge is valuable because the *official* manuals only tell you which buttons to press — they say nothing about how to diagnose a sour shot, how to "temperature surf" a single-boiler machine, why your puck is channeling, or which mods (PID, Silvia steam wand) are worth doing. That troubleshooting knowledge lives in scattered Reddit threads, forum posts, and YouTube comments, and a beginner has to read dozens of them to assemble a coherent mental model. A retrieval system that consolidates this community knowledge and answers a specific question with grounded, attributed advice is genuinely useful.

---

## Documents

The corpus is 12 markdown documents, each covering one subtopic, written to reflect the consensus advice found in the listed communities (content is paraphrased/synthesized from these public sources rather than scraped verbatim, to avoid copyright issues). Each file names its representative source at the top.

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | r/espresso wiki | Beginner dialing-in guide: sour/bitter loop, 1:2 ratio | documents/01_dialing_in_basics.md |
| 2 | Home-Barista.com forum | Sour vs bitter taste troubleshooting | documents/02_grind_taste_troubleshooting.md |
| 3 | r/espresso Bambino megathread | Breville Bambino setup, recipe, baskets | documents/03_breville_bambino.md |
| 4 | Home-Barista.com Gaggia thread | Gaggia Classic temperature surfing + PID mod | documents/04_gaggia_classic_temp_surfing.md |
| 5 | r/gaggiaclassic mod guides | Rancilio Silvia steam wand mod | documents/05_silvia_steam_wand_mod.md |
| 6 | r/espresso puck-prep threads | Channeling, WDT, puck preparation | documents/06_channeling_wdt_puck_prep.md |
| 7 | r/espresso / barista forums | Tamping technique (level > pressure) | documents/07_tamping_technique.md |
| 8 | r/latteart | Milk steaming and microfoam | documents/08_milk_steaming_microfoam.md |
| 9 | r/espresso / roaster blogs | Bean freshness, degassing, resting window | documents/09_bean_freshness_degassing.md |
| 10 | Home-Barista.com | Backflushing, cleaning, descaling | documents/10_backflushing_maintenance.md |
| 11 | r/espresso explainers | Pressurized vs non-pressurized baskets | documents/11_pressurized_vs_nonpressurized.md |
| 12 | Home-Barista.com water threads | Water quality for espresso | documents/12_water_quality.md |

---

## Chunking Strategy

<!-- How will you split documents into chunks? -->

**Chunk size:** ~600 characters per chunk.

**Overlap:** 100 characters.

**Reasoning:** These documents are short, topic-focused guides (roughly 1,500–2,500 characters each) made of self-contained tips and numbered steps. A ~600-character chunk is large enough to hold a complete idea (e.g., the full "sour = under-extracted → grind finer" explanation, or a whole numbered procedure step plus its rationale) but small enough that an embedding stays sharply about one concept rather than averaging several. The 100-character overlap (~1–2 sentences) keeps a tip from being cut in half across a boundary — e.g., a fix and the symptom it addresses staying connected — which directly mitigates the channeling/boundary risk noted below. We split on paragraph/sentence boundaries where possible rather than mid-word.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`. It is small (~80MB), fast on CPU, runs locally for free, and produces 384-dim embeddings that perform well on short-passage semantic similarity — a good fit for a beginner corpus of short tips.

**Top-k:** 4 chunks per query. Enough to assemble a complete answer (a question like "how do I steam milk" may pull from the steaming doc plus the steam-wand-mod doc) without flooding the prompt with marginally relevant text that dilutes grounding.

**Production tradeoff reflection:** If cost were no object and this served real users, I'd weigh: (1) **Accuracy on domain text** — a larger model like `text-embedding-3-large` or a coffee/espresso-domain-tuned model would better disambiguate near-synonyms ("sour" vs "bitter", "pressurized" vs "non-pressurized") that this domain hinges on. (2) **Context length** — MiniLM truncates at 256 tokens, fine for these short docs but limiting if I later ingest long forum threads; a long-context embedder would let me use bigger chunks. (3) **Multilingual support** — the espresso community is global; a multilingual model would let me ingest Italian/German forum content. (4) **Latency & hosting** — local MiniLM has zero network latency and no per-call cost, whereas an API embedder adds latency and cost but offloads compute. For this project the local model wins on cost and simplicity; at scale I'd likely move to a hosted larger model for the retrieval-quality gain and add a reranker.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | My espresso shot pulls really fast and tastes sour. What should I change? | The shot is under-extracted; grind finer (one step at a time) so water meets more resistance and extracts more. |
| 2 | What dose and ratio should I use for a double shot on a Breville Bambino? | About 18 g in to 36 g out (1:2 ratio) in 25–30 seconds, using the single-wall basket. |
| 3 | The steam on my older Gaggia Classic is weak and bubbly. How do I get better microfoam? | Wait for the steam-ready light and purge first; the popular fix is the Rancilio Silvia steam wand mod with a single-hole tip for a tighter whirlpool. |
| 4 | My shots spray everywhere and taste both harsh and sour. How do I fix channeling? | Improve puck prep — use WDT to break up clumps, a dosing funnel, level distribution, and a level tamp; check dose/grind if it persists. |
| 5 | What brew water temperature should I target, and how do I hit it on a Gaggia Classic without a PID? | About 92–96°C (198–205°F); temperature surf — heat fully, wait for the heating light to cycle off, flush, and pull right after it clicks off. |

---

## Anticipated Challenges

1. **Sour/bitter confusability in embedding space.** "Sour" and "bitter" are opposite problems with opposite fixes, but they're textually similar (both are negative taste complaints appearing in the same documents). A query about a sour shot could retrieve the bitter-shot fix, producing exactly the wrong advice. Mitigation: documents state symptom→diagnosis→fix explicitly within each chunk, and top-k=4 gives the generator enough context to pick the matching branch.

2. **Information split across chunk boundaries.** Numbered procedures (temperature surfing, milk steaming) lose meaning if a step is severed from its rationale or from the preceding step. Mitigation: 100-char overlap and sentence-boundary splitting keep adjacent steps connected; choosing 600 chars keeps most single tips whole.

3. **Off-topic / out-of-corpus questions.** Users may ask about machines or topics the corpus doesn't cover (e.g., a specific grinder model). The system must say it doesn't know rather than hallucinate. Mitigation: strict grounding system prompt that instructs the model to answer only from retrieved context and to say so when the context is insufficient.

---

## Architecture

```
                       THE UNOFFICIAL GUIDE — RAG PIPELINE

  ┌──────────────────┐   ┌──────────────┐   ┌─────────────────────────┐
  │ 1. INGESTION     │   │ 2. CHUNKING  │   │ 3. EMBED + VECTOR STORE  │
  │ documents/*.md   │──▶│ ~600 chars,  │──▶│ all-MiniLM-L6-v2         │
  │ read text        │   │ 100 overlap  │   │ (sentence-transformers)  │
  │ (ingest.py)      │   │ (chunk.py)   │   │  → ChromaDB (persistent) │
  └──────────────────┘   └──────────────┘   └─────────────────────────┘
                                                        │
                                                        ▼
  ┌──────────────────┐   ┌─────────────────────────┐   ┌──────────────────────┐
  │ 5. GENERATION    │   │ 4. RETRIEVAL            │   │  user query          │
  │ Groq llama-3.x   │◀──│ embed query → Chroma    │◀──│  (Gradio UI / CLI)   │
  │ grounded prompt  │   │ top-k=4 chunks + sources│   │                      │
  │ + citations      │   │ (rag.py)                │   │  (app.py)            │
  │ (rag.py)         │   └─────────────────────────┘   └──────────────────────┘
  │      │           │
  │      ▼           │
  │  answer + cited  │
  │  source files    │
  └──────────────────┘
```

Stage → tool mapping:
- **Ingestion:** Python file reading of `documents/*.md` (and `.pdf` via pdfplumber if present) — `ingest.py`
- **Chunking:** custom `chunk_text()` — `chunk.py`
- **Embedding + Vector store:** `sentence-transformers` (`all-MiniLM-L6-v2`) → `chromadb` persistent client — `ingest.py` / `rag.py`
- **Retrieval:** Chroma similarity query, top-k=4 — `rag.py`
- **Generation:** `groq` chat completion with a grounding system prompt — `rag.py`
- **Interface:** `gradio` chat UI — `app.py`; plus `evaluate.py` to run the 5 test questions.

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:** Use Claude (Claude Code). Input: this planning.md's Chunking Strategy (600 chars / 100 overlap, sentence-boundary splitting) and Architecture sections, plus the requirement that ingestion read every file in `documents/` and attach the source filename as metadata. Expected output: `chunk.py` with a `chunk_text(text, size=600, overlap=100)` function and `ingest.py` that loads docs, chunks them, and writes to ChromaDB. Verify: print the total chunk count and spot-check that no chunk splits mid-word and that overlap is present between consecutive chunks.

**Milestone 4 — Embedding and retrieval:** Use Claude. Input: the Retrieval Approach section (all-MiniLM-L6-v2, top-k=4, ChromaDB persistent store). Expected output: embedding of chunks at ingest time and a `retrieve(query, k=4)` function returning chunk text + source metadata + distances. Verify: run each of the 5 test questions through retrieval only and confirm the expected source document appears in the top-4.

**Milestone 5 — Generation and interface:** Use Claude. Input: the Grounded Generation requirement and the Architecture (Groq model, grounding system prompt, source citation). Expected output: `generate(query, chunks)` building a grounded prompt, plus a Gradio chat interface in `app.py` and an `evaluate.py` harness. Verify: ask an out-of-corpus question and confirm the system declines rather than hallucinates; confirm answers cite their source files; run evaluate.py and fill the README evaluation table.
