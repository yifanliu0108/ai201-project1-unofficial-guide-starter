# The Unofficial Guide - UCSD Data Science

A retrieval-augmented generation (RAG) assistant for UC San Diego data science
students. It answers course-planning and workload questions using a local corpus of
UCSD DSC course notes, official catalog/course-listing references, r/UCSD-style
student advice, and student planning guides, then cites the source documents it used.

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

The domain is unofficial UCSD Data Science course planning: how to sequence DSC
classes, which courses are bottlenecks, what workloads students should expect, how
EASy/prerequisite planning affects enrollment, and how to choose electives for a
technical data science path.

This knowledge is valuable because official pages list requirements and course
titles, but a data science student also needs practical context: which lower-division
classes unlock the upper division, why DSC 80 matters, whether DSC 30 should be
paired with lighter courses, and when not to rely on prerequisite exceptions. That
advice is scattered across the UCSD catalog, course resource listings, public course
sites, and r/UCSD discussions.

## Document Sources

The corpus contains 12 local text documents in `documents/`. Each document has a
source line, representative URL(s), and a local filename. The documents are
paraphrased/synthesized notes built from UCSD official sources and public student
discussion topics rather than copied webpages.

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
| 12 | `documents/12_ucsd_dsc_four_year_strategy.txt` | Student-maintained UCSD DSC planning notes synthesized from catalog/listings/r/UCSD | Four-year or transfer plan, workload pairing, portfolio strategy |

## Document Pipeline

`ingest.py` loads every `.txt`, `.md`, or `.pdf` file in `documents/`, skips
`.gitkeep`, and attaches `source` plus `chunk_index` metadata to each chunk. This
corpus is plain text, so ingestion reads the files directly as UTF-8.

`chunk.py` splits each document before ChromaDB embeds and stores the chunks. The
current UCSD index contains 35 chunks from 12 documents.

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
   `The major builds through a sequence: DSC 10 introduces the field, DSC 20 and DSC 30 build programming and data structures, DSC 40A and DSC 40B give math/theory foundations, and DSC 80 becomes the first big practice-and-application class.`

2. `03_dsc20_30_programming_sequence.txt`, chunk 1:
   `Student advice consistently says not to overload the quarter you take DSC 30 ... The main difficulty is ... the weekly rhythm of programming work.`

3. `05_dsc80_practice_application.txt`, chunk 1:
   `DSC 80 can feel like the continuation of DSC 30, except with more data analysis in Python. Students who are shaky on dataframe operations, debugging, or writing clear explanations should review before the quarter starts.`

4. `06_dsc100_data_management.txt`, chunk 1:
   `Non-majors may need an EASy request, and students trying to take DSC 100 while finishing DSC 40B or DSC 80 should not assume a prerequisite exception will be approved.`

5. `09_dsc140a_ml_foundations.txt`, chunk 1:
   `Review probability and linear algebra before the quarter starts ... The hard part is often not calling a library; it is understanding what the model is assuming.`

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

- `12_ucsd_dsc_four_year_strategy.txt` (distance 0.318) - finish DSC 10, 20, 30,
  40A, 40B, and 80 because they unlock upper division.
- `01_ucsd_dsc_major_catalog.txt` (distance 0.372) - official major structure and
  lower-division requirements.
- `11_enrollment_and_easypass_advice.txt` (distance 0.391) - enrollment bottlenecks
  and prerequisite planning.

Why relevant: the query asks for sequence planning, and the top chunks combine the
student planning guide, official major structure, and enrollment strategy.

Example 2:

Query: `Can I assume an EASy request will let me take DSC 100 while still finishing a prerequisite?`

Top returned chunks:

- `06_dsc100_data_management.txt` (distance 0.298) - do not assume a DSC 100
  prerequisite exception will be approved.
- `01_ucsd_dsc_major_catalog.txt` (distance 0.329) - students must satisfy
  prerequisites for courses.
- `06_dsc100_data_management.txt` (distance 0.370) - ask the department before
  building a schedule around an exception.

Why relevant: all three chunks directly address prerequisites, DSC 100, or the
policy risk behind assuming an exception.

Example 3:

Query: `What should I know before taking DSC 80?`

Top returned chunks:

- `06_dsc100_data_management.txt` (distance 0.324) - distinguishes DSC 100 from DSC
  80 and notes DSC 80's broader carryover.
- `05_dsc80_practice_application.txt` (distance 0.375) - DSC 80 combines Python,
  pandas, cleaning, visualization, and reasoning.
- `06_dsc100_data_management.txt` (distance 0.385) - notes DSC 80's practical
  cleaning and analysis habits.

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
To plan the UCSD DSC lower-division sequence, focus on completing the prerequisites
on time, including DSC 10, DSC 20, DSC 30, DSC 40A, DSC 40B, and DSC 80, as they
unlock the upper-division core. Additionally, expect to take lower-division courses
in programming, data structures, theory foundations, calculus, linear algebra, and
subject-domain courses.

Sources: 01_ucsd_dsc_major_catalog.txt, 11_enrollment_and_easypass_advice.txt, 12_ucsd_dsc_four_year_strategy.txt
```

Question: `Can I assume an EASy request will let me take DSC 100 while still finishing a prerequisite?`

System response:

```text
No, you should not assume a prerequisite exception will be approved. According to
the unofficial advice, students trying to take DSC 100 while finishing DSC 40B or
DSC 80 should not assume a prerequisite exception will be approved. It is
recommended to check the official prerequisite list and ask the department before
building a schedule that depends on an exception.

Sources: 01_ucsd_dsc_major_catalog.txt, 06_dsc100_data_management.txt
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

Assistant: Before taking DSC 80, you should know that it is a course where you will
be assumed to have certain skills, such as being able to clean data, work in
notebooks, reason about messy datasets, and communicate results.

Sources: 05_dsc80_practice_application.txt, 06_dsc100_data_management.txt

_Retrieved from: 05_dsc80_practice_application.txt, 06_dsc100_data_management.txt_
```

## Evaluation Report

Evaluation command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluate.py
```

Results from the UCSD run:

| # | Question | Expected answer | Retrieved chunks | Actual response | Judgment |
|---|----------|-----------------|------------------|-----------------|----------|
| 1 | How should I plan the UCSD DSC lower-division sequence? | Finish DSC 10, DSC 20, DSC 30, DSC 40A, DSC 40B, and DSC 80 on time because they unlock upper division; plan around prerequisites. | `12_ucsd_dsc_four_year_strategy.txt` 0.318, `01_ucsd_dsc_major_catalog.txt` 0.372, `11_enrollment_and_easypass_advice.txt` 0.391 | Listed DSC 10, 20, 30, 40A, 40B, and 80 as the lower-division sequence and noted they unlock upper division. | Accurate |
| 2 | Is DSC 30 a heavy class, and how should I handle it? | Yes, it is a technical programming/data-structures class; do not overload the quarter, start assignments early, and build debugging/runtime habits. | `05_dsc80_practice_application.txt` 0.396, `03_dsc20_30_programming_sequence.txt` 0.413, `09_dsc140a_ml_foundations.txt` 0.417 | Said DSC 30 is technically demanding, covers data structures and algorithms, and students should not overload the quarter, but did not include the start-early/debugging advice. | Partially accurate |
| 3 | What should I know before taking DSC 80? | DSC 80 combines Python/dataframe work, cleaning, visualization, reasoning, and communication; review debugging and dataframe skills. | `06_dsc100_data_management.txt` 0.324, `05_dsc80_practice_application.txt` 0.375, `06_dsc100_data_management.txt` 0.385 | Said DSC 80 assumes cleaning data, notebooks, messy datasets, and communicating results. | Accurate |
| 4 | Can I assume an EASy request will let me take DSC 100 while still finishing a prerequisite? | No. Do not assume a prerequisite exception; check official prerequisites and ask the department before planning around it. | `06_dsc100_data_management.txt` 0.298, `01_ucsd_dsc_major_catalog.txt` 0.329, `06_dsc100_data_management.txt` 0.370 | Said no, do not assume approval; check prerequisites and ask the department. | Accurate |
| 5 | How should I prepare for DSC 140A? | Review probability and linear algebra, especially if DSC 40A/40B were shaky; focus on model assumptions/objectives and avoid pairing with another math-heavy class. | `09_dsc140a_ml_foundations.txt` 0.405, `05_dsc80_practice_application.txt` 0.441, `06_dsc100_data_management.txt` 0.457 | Said to avoid pairing it with another math-heavy class and treat it as a foundation, but omitted probability, linear algebra, and model assumptions. | Partially accurate |

Overall: 3 accurate, 2 partially accurate, plus a correct refusal for an
out-of-scope UCSD dining question.

## Failure Case Analysis

Failure: Q5, `How should I prepare for DSC 140A?`

Expected answer: `09_dsc140a_ml_foundations.txt` says to review probability and
linear algebra before the quarter, especially if DSC 40A/40B felt shaky, and to
focus on model assumptions/objectives rather than only calling libraries.

Actual behavior: retrieval found a DSC 140A chunk, but it was the later chunk about
not treating DSC 140A as a box to check and not pairing it with another math-heavy
class. The retrieved DSC 140A chunk did not include the earlier sentence containing
the most direct prep advice about probability and linear algebra.

Specific cause: this is a retrieval + chunk-boundary failure. The relevant document
was found, but top-k retrieval returned the wrong neighboring chunk inside that
document. Grounding then kept the answer faithful to the context, so the model did
not invent the missing probability/linear algebra advice. A fix would be neighbor
chunk expansion: when one chunk from a document is retrieved, include the previous
and next chunk from the same source, or use a reranker that scores the exact question
against adjacent chunks.

## Spec Reflection

One way the spec helped: the five evaluation questions forced me to test course
sequencing, workload, prerequisite policy, and ML preparation separately instead of
only asking broad "what classes are good?" questions.

One divergence from the plan: I kept top-k at 3 and used `MAX_DISTANCE = 0.55`
because it reliably refused out-of-scope dining questions. The tradeoff is visible
in Q5: stricter retrieval plus no neighbor expansion can miss the best adjacent
chunk inside the correct source document.

## AI Usage Transparency

Instance 1 - corpus and chunking:

I directed AI assistance to convert the project domain from a fictional university
to UCSD Data Science and to create 12 local source documents around UCSD DSC course
planning. I reviewed and revised the corpus so each file named a source type,
representative URL, and concrete student-facing advice rather than generic college
tips.

Instance 2 - retrieval/generation/evaluation:

I directed AI assistance to update `rag.py`, `app.py`, and `evaluate.py` for UCSD
DSC questions. I reviewed actual evaluation output and kept an honest partial
failure for DSC 140A instead of tuning the test set until every answer looked
perfect. I also kept deterministic source-line replacement in `rag.py` so citations
come from retrieved metadata, not from the model's formatting.
