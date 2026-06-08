# The Unofficial Guide - UCSD Data Science

A retrieval-augmented generation (RAG) assistant I built around the UCSD Data Science
major. The questions I cared about were the ones I would actually ask while planning
quarters: what to take before DSC 80, whether DSC 30 needs room in the schedule, and
when an EASy/prerequisite assumption is risky. The assistant answers from local source
cards and cites the files it used.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # add GROQ_API_KEY from https://console.groq.com
python ingest.py              # builds chroma_db/ from documents/
python evaluate.py            # runs the 5 test questions
python app.py                 # launches the Gradio query UI
```

If the embedding model is already cached locally:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluate.py
```

Pipeline files:

- `chunk.py` - sentence-aware chunking
- `ingest.py` - document loading, embedding, and ChromaDB indexing
- `rag.py` - retrieval, grounded generation, source enforcement
- `app.py` - Gradio query interface
- `evaluate.py` - evaluation harness

## Domain

The domain is unofficial UCSD Data Science course planning: sequencing DSC classes,
spotting bottlenecks, understanding workload, handling EASy/prerequisite uncertainty,
and choosing electives for a more technical data science path.

This knowledge is hard to get from one official page. The catalog tells me what is
required, but it does not fully answer questions like "what class unlocks the rest of
my plan?" or "am I making my quarter miserable by pairing these?" I used the UCSD
catalog/course listings as the official base and added short student-facing source
cards for the planning advice that usually lives in r/UCSD posts and peer notes.

## Document Sources

The corpus contains 12 local text documents in `documents/`. I wrote them as source
cards: each file says where the information came from, why I kept it, and what the
planning takeaway is. I did this instead of scraping whole pages because the project
only needs a small, inspectable corpus, and I wanted every source file to be readable
when I debug retrieval.

| # | File | Source description | What it covers |
|---|------|--------------------|----------------|
| 1 | `documents/01_ucsd_dsc_major_catalog.txt` | UCSD General Catalog Data Science B.S. - https://catalog.ucsd.edu/curric/DSC-ug.html | Major structure, lower division, selective/capped status, prerequisites |
| 2 | `documents/02_dsc10_student_guide.txt` | UCSD DSC course listings and r/UCSD starting-major advice - https://courses.ucsd.edu/courselist.aspx?name=DSC | DSC 10 workflow, notebooks, early data science habits |
| 3 | `documents/03_dsc20_30_programming_sequence.txt` | UCSD DSC listings and r/UCSD DSC 20/30 workload discussions | Programming sequence, DSC 30 workload, debugging habits |
| 4 | `documents/04_dsc40a_40b_theory_foundations.txt` | UCSD DSC listings and student planning notes | Theory foundations, probability/modeling/math reasoning |
| 5 | `documents/05_dsc80_practice_application.txt` | DSC 80 public site and r/UCSD DSC 80 discussions - https://dsc80.com/ | DSC 80 as a practice/application checkpoint |
| 6 | `documents/06_dsc100_data_management.txt` | UCSD DSC listings and r/UCSD DSC 100 prerequisite discussions | Data management, EASy/prerequisite caution |
| 7 | `documents/07_dsc102_scalable_analytics.txt` | UCSD DSC listings and systems/scalable analytics advice | DSC 102 systems workload and preparation |
| 8 | `documents/08_dsc106_visualization.txt` | UCSD DSC listings and visualization project notes | DSC 106 chart design, communication, portfolio value |
| 9 | `documents/09_dsc140a_ml_foundations.txt` | UCSD DSC listings and r/UCSD DSC major discussions | DSC 140A ML foundations, probability, linear algebra |
| 10 | `documents/10_cse_151a_and_dsc_electives.txt` | r/UCSD data science program discussions and UCSD listings | CSE 151A, DSC electives, technical direction |
| 11 | `documents/11_enrollment_and_easypass_advice.txt` | r/UCSD enrollment, second pass, and EASy discussions | Enrollment strategy, EASy limits, transfer petitions |
| 12 | `documents/12_ucsd_dsc_four_year_strategy.txt` | My UCSD DSC planning notes compiled from catalog/listings/r/UCSD | Four-year or transfer plan, workload pairing, portfolio strategy |

## Document Pipeline

`ingest.py` loads every `.txt`, `.md`, or `.pdf` file in `documents/`, skips
`.gitkeep`, and attaches `source` plus `chunk_index` metadata to each chunk. This
corpus is plain text, so ingestion reads the files directly as UTF-8.

`chunk.py` splits each document before ChromaDB embeds and stores the chunks. The
current UCSD index contains 31 chunks from 12 documents.

## Chunking Strategy

Chunk size is about 600 characters with 100 characters of overlap.

This fits the UCSD DSC corpus because most documents are compact planning notes: one
paragraph usually explains a single course, enrollment issue, or workload warning.
Six hundred characters is enough to keep a course name, the advice, and the reason
together. It is also small enough that a chunk about DSC 80 does not blur into DSC
100 or DSC 140A. The 100-character overlap protects boundary cases where the first
sentence of the next chunk depends on the end of the previous chunk.

The splitter prefers paragraph and sentence boundaries. If a sentence is longer than
the target size, it wraps on whitespace rather than cutting words.

### Sample Chunks

1. `01_ucsd_dsc_major_catalog.txt`, chunk 1:
   `Facts from the catalog: - The major is 112 units total: 52 lower-division units and 60 upper-division units ... The lower-division DSC sequence includes COGS 9, DSC 10/R, DSC 20/R, DSC 30/R, DSC 40A/R, DSC 40B/R, and DSC 80/R.`

2. `03_dsc20_30_programming_sequence.txt`, chunk 1:
   `Student advice I would actually use: do not overload the DSC 30 quarter unless coding is already comfortable. Start assignments when they come out ...`

3. `05_dsc80_practice_application.txt`, chunk 1:
   `Student thread note: people talk about DSC 80 like the continuation of DSC 30, but with more data analysis in Python. That means dataframe operations, debugging, and clear written explanations all matter.`

4. `06_dsc100_data_management.txt`, chunk 1:
   `Enrollment note: do not build a schedule assuming an EASy request will override a DSC 100 prerequisite. If someone is still finishing DSC 40B or DSC 80, they should check the official prerequisite list ...`

5. `09_dsc140a_ml_foundations.txt`, chunk 1:
   `Student prep note: review probability and linear algebra before the quarter starts, especially if DSC 40A/40B felt shaky. The point is not only calling a model ...`

## Retrieval

Embedding model: `all-MiniLM-L6-v2` from `sentence-transformers`.

Vector store: persistent ChromaDB collection at `chroma_db/`.

Top-k: 3 chunks per query.

Relevance filtering: chunks with cosine distance above `MAX_DISTANCE = 0.55` are
dropped so weak student-life matches do not reach the generator.

### Production Model Tradeoffs

For this project, `all-MiniLM-L6-v2` is local, free, fast, and good enough for short
student-advice passages. For production I would weigh:

- Accuracy on UCSD-specific terms such as `EASy`, `DSC 80`, `DSC 40B`, `CSE 151A`,
  and `second pass`.
- Context length, because full Reddit threads and course pages are longer than this
  starter corpus.
- Latency and cost of local embeddings versus hosted API embeddings.
- Whether a hybrid keyword + semantic retriever is needed for exact course numbers.
- Whether a reranker should pull the best neighboring chunk when the first hit has
  only part of the answer.

### Retrieval Test Examples

Example 1:

Query: `How should I plan the UCSD DSC lower-division sequence?`

Top returned chunks:

- `12_ucsd_dsc_four_year_strategy.txt` (distance 0.352) - finish DSC 10, 20, 30,
  40A, 40B, and 80 because they unlock upper division.
- `03_dsc20_30_programming_sequence.txt` (distance 0.390) - DSC 20/30 are the
  programming and data-structures base.
- `01_ucsd_dsc_major_catalog.txt` (distance 0.397) - official major structure and
  lower-division requirements.

Why relevant: the query asks for sequence planning, and the top chunks combine the
student planning guide, programming-sequence notes, and official major structure.

Example 2:

Query: `Can I assume an EASy request will let me take DSC 100 while still finishing a prerequisite?`

Top returned chunks:

- `06_dsc100_data_management.txt` (distance 0.326) - do not assume a DSC 100
  prerequisite exception will be approved.
- `06_dsc100_data_management.txt` (distance 0.389) - ask the department before
  building a schedule around an exception.
- `02_dsc10_student_guide.txt` (distance 0.458) - explains early DSC habits that
  later classes build on.

Why relevant: the top DSC 100 chunks directly address prerequisites and the policy
risk behind assuming an exception.

Example 3:

Query: `What should I know before taking DSC 80?`

Top returned chunks:

- `05_dsc80_practice_application.txt` (distance 0.323) - DSC 80 is a practical
  bridge into upper division.
- `06_dsc100_data_management.txt` (distance 0.368) - distinguishes DSC 80 from DSC
  100 and explains its carryover.
- `02_dsc10_student_guide.txt` (distance 0.422) - notebook/tabular data habits from
  DSC 10 feed into DSC 80.

## Grounded Generation

`rag.py` builds numbered context passages with source filenames:

```text
[1] (source: 05_dsc80_practice_application.txt)
...
```

The system prompt tells the model to answer only from provided context, avoid
inventing UCSD course numbers, prerequisites, policies, or instructors, and say:

```text
I don't have enough information on that in my sources.
```

when context is insufficient.

Grounding is enforced by:

- filtering chunks above cosine distance `0.55`;
- returning the refusal message without calling the LLM when retrieval is empty;
- replacing any model-written `Sources:` line with the actual retrieved source
  filenames.

### Example Grounded Responses

Question: `How should I plan the UCSD DSC lower-division sequence?`

System response:

```text
To plan the UCSD DSC lower-division sequence, you should build your plan around
prerequisites and finish DSC 10, DSC 20, DSC 30, DSC 40A, DSC 40B, and DSC 80 on time
because they unlock the upper-division core. A delay in one lower-division class can
cascade into upper-division courses. It's also important to balance your workload and
not stack too many heavy programming, theory, or project courses in one quarter.

Sources: 01_ucsd_dsc_major_catalog.txt, 03_dsc20_30_programming_sequence.txt, 12_ucsd_dsc_four_year_strategy.txt
```

Question: `Can I assume an EASy request will let me take DSC 100 while still finishing a prerequisite?`

System response:

```text
No, you should not assume an EASy request will override a DSC 100 prerequisite.
According to the course note, "do not build a schedule assuming an EASy request will
override a DSC 100 prerequisite." If you are still finishing DSC 40B or DSC 80, you
should check the official prerequisite list and ask the department before depending
on an exception.

Sources: 02_dsc10_student_guide.txt, 06_dsc100_data_management.txt
```

Out-of-scope question:

```text
Q: Which UCSD dining hall has the best late-night food?
A: I don't have enough information on that in my sources.
```

## Query Interface

The interface is a Gradio chat UI in `app.py`.

Input field: a plain-language UCSD data science question, such as:

```text
What should I know before taking DSC 80?
```

Output fields:

- a grounded answer;
- a deterministic `Sources:` line from `rag.answer()`;
- a `_Retrieved from: ..._` line in the UI.

Sample interaction transcript:

```text
User: What should I know before taking DSC 80?

Assistant: To take DSC 80, you should have a solid foundation in the data science
lifecycle, algorithms, statistics, visualization, and machine learning ideas, as it
pulls these concepts together. Additionally, having a strong grasp of the notebook
workflow, Jupyter, and tabular data from DSC 10 is crucial, as it will make DSC 80
and subsequent courses feel less challenging.

Sources: 02_dsc10_student_guide.txt, 05_dsc80_practice_application.txt, 06_dsc100_data_management.txt

_Retrieved from: 02_dsc10_student_guide.txt, 05_dsc80_practice_application.txt, 06_dsc100_data_management.txt_
```

## Evaluation Report

Evaluation command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluate.py
```

Results from the UCSD run:

| # | Question | Expected answer | Retrieved chunks | Actual response | Judgment |
|---|----------|-----------------|------------------|-----------------|----------|
| 1 | How should I plan the UCSD DSC lower-division sequence? | Finish DSC 10, DSC 20, DSC 30, DSC 40A, DSC 40B, and DSC 80 on time because they unlock upper division; plan around prerequisites. | `12_ucsd_dsc_four_year_strategy.txt` 0.352, `03_dsc20_30_programming_sequence.txt` 0.397, `01_ucsd_dsc_major_catalog.txt` 0.397 | Said to build around prerequisites, finish DSC 10/20/30/40A/40B/80 on time, and avoid stacking too many heavy classes. | Accurate |
| 2 | Is DSC 30 a heavy class, and how should I handle it? | Yes, it is a technical programming/data-structures class; do not overload the quarter, start assignments early, and build debugging/runtime habits. | `02_dsc10_student_guide.txt` 0.301, `07_dsc102_scalable_analytics.txt` 0.388, `03_dsc20_30_programming_sequence.txt` 0.427 | Said DSC 30 is challenging, exposes weak programming habits, and needs a solid DSC 20 foundation, but omitted start-early/debugging advice. | Partially accurate |
| 3 | What should I know before taking DSC 80? | DSC 80 combines Python/dataframe work, cleaning, visualization, reasoning, and communication; review debugging and dataframe skills. | `05_dsc80_practice_application.txt` 0.323, `06_dsc100_data_management.txt` 0.368, `02_dsc10_student_guide.txt` 0.422 | Said DSC 80 bridges lower division into upper division, pulls together the data science lifecycle, and requires notebook-workflow comfort. | Accurate |
| 4 | Can I assume an EASy request will let me take DSC 100 while still finishing a prerequisite? | No. Do not assume a prerequisite exception; check official prerequisites and ask the department before planning around it. | `06_dsc100_data_management.txt` 0.326, `06_dsc100_data_management.txt` 0.389, `02_dsc10_student_guide.txt` 0.458 | Said no, do not assume EASy will override the DSC 100 prerequisite. | Accurate |
| 5 | How should I prepare for DSC 140A? | Review probability and linear algebra, especially if DSC 40A/40B were shaky; focus on model assumptions/objectives and avoid pairing with another math-heavy class. | `06_dsc100_data_management.txt` 0.428, `02_dsc10_student_guide.txt` 0.450, `05_dsc80_practice_application.txt` 0.452 | Refused: `I don't have enough information on that in my sources.` | Inaccurate |

Overall: 3 accurate, 1 partially accurate, 1 inaccurate, plus a correct refusal for an
out-of-scope UCSD dining question.

## Failure Case Analysis

Failure: Q5, `How should I prepare for DSC 140A?`

Expected answer: `09_dsc140a_ml_foundations.txt` says to review probability and
linear algebra before the quarter, especially if DSC 40A/40B felt shaky, and to
focus on model assumptions/objectives rather than only calling libraries.

Actual behavior: retrieval did not return `09_dsc140a_ml_foundations.txt` at all.
The top chunks were DSC 100, DSC 10, and DSC 80, so the model refused instead of
answering from unrelated context.

Specific cause: this is a retrieval failure caused by the phrasing of the query and
the small corpus. The relevant DSC 140A source exists, but the semantic retriever
matched the broad "prepare for a DSC class" wording to general preparation chunks
rather than the course-number-specific DSC 140A chunk. A fix would be hybrid search
or course-code boosting so exact strings like `DSC 140A` receive more weight than
general preparation language.

## Spec Reflection

One way the spec helped: the five evaluation questions forced me to test course
sequencing, workload, prerequisite policy, and ML preparation separately instead of
only asking broad "what classes are good?" questions.

One divergence from the plan: I kept top-k at 3 and used `MAX_DISTANCE = 0.55`
because it reliably refused out-of-scope dining questions. The tradeoff is visible
in Q5: strict semantic retrieval missed the DSC 140A source when the query wording
looked like general preparation advice.

## AI Usage Transparency

Instance 1 - corpus and chunking:

I used Codex after choosing the UCSD Data Science domain myself. I asked it to help
turn my source list and chunking plan into the initial document/chunking setup, then I
rewrote the 12 documents as my own source cards. I kept the official catalog facts
concrete and edited the advice sections so they read like notes I would actually use
while planning quarters.

Instance 2 - retrieval/generation/evaluation:

I used Codex to wire the retrieval, Gradio interface, and evaluation harness together.
I checked the actual outputs myself, kept the DSC 140A failure instead of hiding it,
and added deterministic source-line replacement in `rag.py` after seeing that citation
formatting should come from metadata rather than model wording.
