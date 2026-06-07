# The Unofficial Guide — Project 1

A retrieval-augmented generation (RAG) assistant that answers home-espresso
questions using consolidated community knowledge, grounded in an indexed document
corpus and citing its sources.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt "gradio>=4.0.0"

cp .env.example .env          # then paste your free key from https://console.groq.com
python ingest.py              # build the ChromaDB index from documents/
python evaluate.py            # run the 5 test questions
python app.py                 # launch the Gradio chat UI
```

Pipeline files: `chunk.py` (chunking) · `ingest.py` (ingestion + embedding + vector
store) · `rag.py` (retrieval + grounded generation) · `app.py` (Gradio interface) ·
`evaluate.py` (evaluation harness).

---

## Domain

This system covers **practical home-espresso technique for entry-level machines**
(Breville Bambino, Gaggia Classic Pro, and similar) — how to dial in a shot,
diagnose taste problems, prep the puck, steam milk, maintain the machine, and
choose worthwhile mods.

This knowledge is valuable and hard to find officially because manufacturer
manuals only document *operation* ("press this button"), not *technique*. The
real know-how — why a fast shot tastes sour, how to "temperature surf" a
single-boiler machine, what causes channeling, whether the Silvia steam-wand mod
is worth it — lives scattered across thousands of Reddit (r/espresso,
r/gaggiaclassic) threads and Home-Barista forum posts. A beginner has to read
dozens of conflicting posts to assemble a coherent mental model. This system
consolidates that into one place and answers a specific question with grounded,
attributed advice.

---

## Document Sources

12 documents, each a topic-focused guide reflecting the consensus advice in the
listed communities. Content is **paraphrased/synthesized** from these public
sources (not scraped verbatim) to avoid copyright issues; each file names its
representative source at the top.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | r/espresso "Start Here" wiki | Reddit wiki | documents/01_dialing_in_basics.md |
| 2 | Home-Barista.com forum thread "Why does my espresso taste sour?" | Forum thread | documents/02_grind_taste_troubleshooting.md |
| 3 | r/espresso Breville Bambino megathread | Reddit megathread | documents/03_breville_bambino.md |
| 4 | Home-Barista.com Gaggia Classic owners thread | Forum thread | documents/04_gaggia_classic_temp_surfing.md |
| 5 | r/gaggiaclassic Silvia-wand mod guides | Reddit guide | documents/05_silvia_steam_wand_mod.md |
| 6 | r/espresso puck-prep / channeling discussions | Reddit threads | documents/06_channeling_wdt_puck_prep.md |
| 7 | r/espresso & barista forum tamping consensus | Reddit/forum | documents/07_tamping_technique.md |
| 8 | r/latteart beginner milk-steaming guide | Reddit guide | documents/08_milk_steaming_microfoam.md |
| 9 | r/espresso & roaster blogs on bean freshness | Reddit/blog | documents/09_bean_freshness_degassing.md |
| 10 | Home-Barista.com maintenance threads | Forum threads | documents/10_backflushing_maintenance.md |
| 11 | r/espresso pressurized-vs-non basket explainers | Reddit threads | documents/11_pressurized_vs_nonpressurized.md |
| 12 | Home-Barista.com water-chemistry threads | Forum threads | documents/12_water_quality.md |

The sources deliberately span different subtopics (extraction theory, machine-specific
setup, hardware mods, puck prep, milk, beans, maintenance, water) and different
communities (Reddit subreddits and the Home-Barista forum) for coverage and
perspective.

---

## Chunking Strategy

**Chunk size:** ~600 characters (`CHUNK_SIZE` in `chunk.py`).

**Overlap:** 100 characters.

**Why these choices fit your documents:** The documents are short, topic-focused
guides (~1,500–2,500 characters each) built from self-contained tips and numbered
procedures. 600 characters is large enough to hold one complete idea — the full
"sour = under-extracted → grind finer" explanation, or a procedure step *with* its
rationale — but small enough that each embedding stays sharply about a single
concept instead of averaging several. The 100-character overlap (~1–2 sentences)
prevents a tip from being severed from the symptom it addresses across a boundary.
Preprocessing: chunking splits on paragraph and sentence boundaries (never
mid-word); an over-long sentence is hard-wrapped on whitespace. The source filename
is stripped to metadata, not embedded into the chunk text.

**Final chunk count:** **43 chunks** across the 12 documents (3–4 per document).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim, runs
locally on CPU). Chosen because it's small (~80MB), fast, free, and strong on
short-passage semantic similarity — a good match for a corpus of short tips, and
it requires no API key or network for the retrieval half of the pipeline.

**Production tradeoff reflection:** If I were deploying this for real users and
cost weren't a constraint, I'd weigh:
- **Accuracy on domain-specific text** — a larger model (e.g. OpenAI
  `text-embedding-3-large`) or a domain-tuned embedder would better separate the
  near-synonyms this domain hinges on ("sour" vs "bitter", "pressurized" vs
  "non-pressurized"). I'd likely also add a cross-encoder **reranker** over the
  top-k.
- **Context length** — MiniLM truncates around 256 tokens. Fine for these short
  docs, but if I ingested long forum threads I'd want a long-context embedder so I
  could use larger chunks without truncation.
- **Multilingual support** — the espresso community is global; a multilingual model
  would let me ingest Italian/German forum content the English model can't embed
  well.
- **Latency vs. local hosting** — local MiniLM has zero network latency and zero
  per-call cost; an API embedder adds both but offloads compute and scales. For a
  single-user project local wins; at scale I'd move to a hosted larger model.

---

## Grounded Generation

**System prompt grounding instruction (verbatim from `rag.py`):** the model is
told it is "The Unofficial Guide" and that it *must answer ONLY using the numbered
CONTEXT passages provided*. The explicit rules are:

- "Base every claim on the CONTEXT. Do not use outside knowledge or invent facts,
  numbers, model names, or steps that are not in the CONTEXT."
- "If the CONTEXT does not contain enough information to answer, say plainly: *I
  don't have information about that in my sources.* Do not guess."
- "Cite the source file(s) you used at the end of your answer."

**Structural choices that enforce grounding (not just the prompt):**
1. **Distance filtering** — `retrieve()` drops chunks whose cosine distance exceeds
   `MAX_DISTANCE`, so weakly-related context never reaches the model.
2. **If retrieval returns nothing, `generate()` short-circuits** and returns the
   "I don't have information" message without ever calling the LLM — the model
   cannot answer from a blank context.
3. **Numbered, source-labeled context** — each passage is injected as
   `[n] (source: filename) …`, giving the model an unambiguous, citable handle on
   each fact.
4. **Low temperature (0.2)** to keep the model close to the provided text.

**How source attribution is surfaced in the response:** the model cites the source
filename(s) inline at the end of its answer, and the interface (`app.py`)
additionally appends a "_Retrieved from: …_" line listing every source document
that contributed to the retrieved context, so the user can verify the answer
against the originals.

---

## Evaluation Report

Results below are from an actual `python evaluate.py` run (Groq `llama-3.3-70b-versatile`).
Retrieval quality reflects the top-4 retrieved source files and cosine distances
the harness printed; system responses are summarized from the same run.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Shot pulls fast and tastes sour — what to change? | Under-extracted → grind finer | "Grind finer so the water meets more resistance and spends more time extracting." Cited 01, 02. | Relevant (01 @ 0.248, 02 @ 0.298) | Accurate |
| 2 | Dose & ratio for a double on a Breville Bambino? | ~18 g in → 36 g out, 1:2, 25–30 s | "18 g in, 36 g out, 1:2 ratio, ~25–30 s." Cited 01, 03. | Relevant (03 @ 0.454/0.509) | Accurate |
| 3 | Weak/bubbly steam on old Gaggia Classic — better microfoam? | Wait for steam light + purge; Silvia wand single-hole mod | "Wait for the steam-ready light, purge the wand, stretch with the tip near the surface then submerge to make a whirlpool." Cited 05, 08. | Relevant (05 @ 0.307, 08 @ 0.436) | Partially accurate — correct technique, but it omitted the Rancilio Silvia single-hole wand *mod* (the hardware answer), drawing mostly from the milk-steaming doc. |
| 4 | Shots spray; taste harsh AND sour — fix channeling? | WDT, funnel, level distribution + tamp | "Channeling comes from uneven density/clumps and bad distribution — *but the context does not provide specific steps to fix it.*" Did **not** give the WDT fix. | Partially relevant — only 1 of 4 chunks was from 06 (the channeling doc, @ 0.499 rank 3); the other 3 were the taste doc 02. | **Inaccurate / incomplete** — diagnosed the cause but failed to give the fix (see Failure Case). |
| 5 | Brew temp target & hitting it on Gaggia without a PID? | ~92–96°C; temperature surf | "92–96°C (198–205°F); temperature-surf: heat fully until the light is off, wait 30–60 s, flush, wait for the light to cycle, pull right after it clicks off." Cited 04. | Relevant (04 @ 0.260/0.272/0.446) | Accurate |

**Grounding check (out-of-corpus):** asked about Niche Zero burr size / seasoning —
the closest chunks were weakly related (≥0.604) and contained no answer, and the
system correctly replied "I don't have information about that in my sources." ✅

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

Overall: 3 of 5 fully accurate, 1 partially accurate (Q3), 1 incomplete (Q4), and a
correct refusal on the out-of-corpus question — an honest result that exposes a real
chunk-boundary weakness rather than a suspiciously perfect score.

---

## Failure Case Analysis

**Question that failed:** Q4 — "My shots spray everywhere and taste both harsh and
sour. How do I fix channeling?"

**What the system returned:** It correctly identified the *cause* — "uneven density
in the puck, usually due to clumps from the grinder and uneven distribution" — but
then said: "The CONTEXT does not provide specific steps to fix channeling beyond
identifying its root cause." It never gave the actual fix (WDT, dosing funnel, level
distribution, level tamp), even though that fix exists in the corpus in
`06_channeling_wdt_puck_prep.md`.

**Root cause (tied to a specific pipeline stage):** A **retrieval + chunk-boundary**
failure, and notably *not* a hallucination — grounding worked exactly as designed.
Two things compounded:
1. The query is phrased around *taste* ("harsh and sour"), so embedding similarity
   was dominated by the taste-troubleshooting doc `02`, which took 3 of the 4 slots
   (ranks 1, 2, 4). Only **one** chunk from the correct doc `06` was retrieved
   (rank 3, distance 0.499).
2. That single `06` chunk happened to be the **opening chunk** — the one defining
   what channeling is and its root cause. The numbered *fix* (WDT → funnel →
   distribute → tamp) lives in a **later chunk of the same document** that was never
   retrieved. So the model was handed the cause but not the cure, and — correctly
   obeying the grounding rule — refused to invent the steps it didn't have.

This is precisely the "information split across chunk boundaries" risk predicted in
planning.md: chunking severed the *definition* of channeling from its *fix*, and
retrieval only pulled the former.

**What you would change to fix it:** (1) **Retrieve more than one chunk per matched
document** — e.g., expand a hit to include its neighboring chunks (the next chunk of
`06` holds the fix), or raise top-k. (2) Add a **cross-encoder reranker** so the
channeling doc outranks the taste doc. (3) For this short corpus specifically,
**chunk by whole document or whole section** so a tip's symptom, cause, and fix stay
in one retrievable unit — the cleanest fix here, since the documents are small.

---

## Spec Reflection

**One way the spec helped you during implementation:** Writing the Chunking
Strategy and Architecture in planning.md *before* coding meant `chunk.py` and
`ingest.py` were written to a fixed contract (600/100, sentence-boundary splitting,
source-as-metadata) rather than being improvised. When I implemented retrieval, the
Evaluation Plan's five questions were already written, so I could validate the
retrieval stage immediately against expected source documents — and I caught the Q4
channeling ranking issue early because the spec had predicted exactly that
sour/bitter confusability risk.

**One way your implementation diverged from the spec, and why:** The spec didn't
mention a relevance threshold; while implementing grounding I added `MAX_DISTANCE`
filtering plus a short-circuit that returns the "I don't have information" message
without calling the LLM when nothing is retrieved. I added this because the
Anticipated Challenges section flagged out-of-corpus questions, and I realized the
system prompt alone is a soft guarantee — a structural filter makes the decline
behavior deterministic rather than relying on the model to behave.

---

## AI Usage

**Instance 1 — Chunking implementation**

- *What I gave the AI:* the Chunking Strategy section of planning.md (600-char
  chunks, 100-char overlap, sentence-boundary splitting, no mid-word cuts) and asked
  it to implement `chunk_text()`.
- *What it produced:* a first version that sliced the string at fixed 600-character
  offsets — fast, but it cut sentences (and once a word) in half.
- *What I changed or overrode:* I rejected the fixed-offset approach and directed it
  to split into sentences first and *assemble* chunks up to the size limit, carrying
  a 100-char overlap tail between chunks, plus a hard-wrap fallback for any single
  sentence longer than the chunk size. This matches the spec's "never cut mid-word"
  requirement.

**Instance 2 — Grounding mechanism**

- *What I gave the AI:* the Grounded Generation requirement and asked it to write the
  system prompt and `generate()`.
- *What it produced:* a reasonable system prompt instructing the model to "use the
  documents," passing all retrieved chunks straight to the LLM.
- *What I changed or overrode:* I judged a prompt alone too soft for the
  out-of-corpus risk from planning.md, so I added structural grounding the AI hadn't:
  a `MAX_DISTANCE` relevance filter, a short-circuit that returns the canned "I don't
  have information" answer *without* calling the LLM when retrieval is empty, and
  numbered source-labeled context blocks to make citations unambiguous. I also tightened
  the prompt to forbid inventing "numbers, model names, or steps," because this domain
  is full of specific figures the model could otherwise hallucinate.
