# Project 1 Planning: The Unofficial Guide

## Domain

The domain is unofficial course-planning knowledge for UC San Diego data science
students. I am focusing on the DSC major sequence: lower-division setup, workload
pairing, core upper-division courses, EASy/prerequisite planning, and elective
strategy.

This knowledge is valuable because UCSD official pages tell students what courses
exist and what requirements must be completed, but they do not fully answer the
student version of the problem: "What should I take first?", "Which classes unlock
the rest of my plan?", "Can I rely on a prerequisite exception?", and "What should I
review before a hard DSC class?" That practical knowledge is split across the UCSD
catalog, the course resource listing, public course sites, r/UCSD threads, and
student-maintained planning notes.

## Documents

The corpus has 12 local text documents. Each document names a source type,
representative URL(s), and a local file path.

| # | File | Specific source |
|---|------|-----------------|
| 1 | `documents/01_ucsd_dsc_major_catalog.txt` | UCSD General Catalog Data Science B.S. page: https://catalog.ucsd.edu/curric/DSC-ug.html |
| 2 | `documents/02_dsc10_student_guide.txt` | UCSD DSC course listing plus r/UCSD starting-major advice: https://courses.ucsd.edu/courselist.aspx?name=DSC |
| 3 | `documents/03_dsc20_30_programming_sequence.txt` | UCSD DSC course listing plus r/UCSD DSC 20/30 workload discussions |
| 4 | `documents/04_dsc40a_40b_theory_foundations.txt` | UCSD DSC course listing and student planning notes for DSC 40A/40B |
| 5 | `documents/05_dsc80_practice_application.txt` | DSC 80 public course site and r/UCSD DSC 80 discussion: https://dsc80.com/ |
| 6 | `documents/06_dsc100_data_management.txt` | UCSD DSC listings and r/UCSD DSC 100 prerequisite/enrollment discussions |
| 7 | `documents/07_dsc102_scalable_analytics.txt` | UCSD DSC listings and student advice about scalable analytics/systems |
| 8 | `documents/08_dsc106_visualization.txt` | UCSD DSC listings and student notes on data visualization projects |
| 9 | `documents/09_dsc140a_ml_foundations.txt` | UCSD DSC listings and r/UCSD data science major discussions about ML foundations |
| 10 | `documents/10_cse_151a_and_dsc_electives.txt` | r/UCSD data science program discussions and UCSD DSC/CSE course listings |
| 11 | `documents/11_enrollment_and_easypass_advice.txt` | r/UCSD enrollment, second pass, and EASy-request threads |
| 12 | `documents/12_ucsd_dsc_four_year_strategy.txt` | Student-maintained UCSD DSC planning notes synthesized from catalog, course listings, and r/UCSD advice |

The sources cover official requirements, individual DSC courses, enrollment
bottlenecks, project workload, theory preparation, and long-term technical direction.

## Chunking Strategy

Chunk size: about 600 characters.

Overlap: 100 characters.

Reasoning: these documents are short course-planning notes rather than long manuals.
Most useful facts fit in a compact paragraph: DSC 30 workload, DSC 80 preparation,
DSC 100 prerequisite risk, or DSC 140A math preparation. A 600-character chunk keeps
the course name, the advice, and the rationale together. Smaller chunks would split
the course from its explanation; much larger chunks would mix multiple DSC courses
and weaken retrieval precision. The 100-character overlap helps when a course name or
warning appears at a boundary.

The implementation should split on paragraph/sentence boundaries and only hard-wrap
on whitespace when a single sentence exceeds the target size. Source filenames should
be stored as metadata for deterministic citations.

## Retrieval Approach

Embedding model: `all-MiniLM-L6-v2` through `sentence-transformers`.

Vector store: ChromaDB persistent collection in `chroma_db/`.

Top-k: 3 chunks.

Relevance threshold: filter out chunks with cosine distance above `0.55`.

Why: the corpus is small and the passages are short, so a local MiniLM model is
fast enough and avoids API cost for retrieval. Top-k 3 usually gives enough context:
one course-specific chunk plus one planning/enrollment chunk plus one supporting
source. The distance threshold is needed because a UCSD data science student may ask
about dining, housing, clubs, or other topics outside the corpus.

Production tradeoff reflection: in production I would test embedding models on UCSD
course-number queries, because exact strings like `DSC 80`, `DSC 100`, `DSC 140A`,
`EASy`, and `CSE 151A` matter. I would weigh cost, latency, context length,
multilingual student posts, and whether a hybrid BM25 + semantic search would rescue
exact course-number matches. I would also add neighbor expansion or reranking so a
hit on one DSC 140A chunk can include adjacent chunks from the same source.

## Evaluation Plan

| # | Test question | Expected correct answer |
|---|---------------|-------------------------|
| 1 | How should I plan the UCSD DSC lower-division sequence? | Finish DSC 10, DSC 20, DSC 30, DSC 40A, DSC 40B, and DSC 80 on time because they unlock upper-division courses; plan around prerequisites. |
| 2 | Is DSC 30 a heavy class, and how should I handle it? | Yes. It is technical programming/data structures; avoid overloading the quarter, start assignments early, and build debugging/runtime habits. |
| 3 | What should I know before taking DSC 80? | DSC 80 combines Python/dataframe work, cleaning, visualization, reasoning, and communication; review dataframe operations, debugging, and explanations. |
| 4 | Can I assume an EASy request will let me take DSC 100 while still finishing a prerequisite? | No. Do not assume a prerequisite exception; check the official prerequisite list and ask the department/advising before planning around it. |
| 5 | How should I prepare for DSC 140A? | Review probability and linear algebra, especially if DSC 40A/40B felt shaky; understand model assumptions/objectives and avoid pairing with another math-heavy class. |

Each expected answer is specific enough to judge as accurate, partially accurate, or
inaccurate.

## Anticipated Challenges

1. Exact course-number retrieval. `DSC 80`, `DSC 100`, and `DSC 140A` are short
strings that carry a lot of meaning. Semantic search can miss an exact course-number
detail if the user phrases the question generally.

2. Neighboring chunk misses. A query may retrieve the right source document but the
wrong chunk inside it. For example, a DSC 140A query might retrieve the scheduling
warning but miss the adjacent chunk that says to review probability and linear
algebra.

3. Policy hallucination risk. Enrollment and EASy rules are high-stakes for a
student schedule. The system must not invent prerequisite exceptions or make claims
about department approval unless the retrieved context says so.

4. Out-of-scope UCSD questions. A data science student might ask about housing,
dining, parking, clubs, or financial aid. The system should refuse rather than answer
from general knowledge.

## Architecture

```text
documents/*.txt
     |
     v
1. Ingestion (ingest.py)
   - read UCSD DSC source notes
   - store source filename and chunk index
     |
     v
2. Chunking (chunk.py)
   - about 600 characters
   - 100-character overlap
   - sentence/paragraph boundaries
     |
     v
3. Embedding + Vector Store (ingest.py)
   - all-MiniLM-L6-v2 via sentence-transformers
   - ChromaDB persistent collection
     |
     v
4. Retrieval (rag.py)
   - embed UCSD student query
   - top-k=3
   - filter distance > 0.55
     |
     v
5. Grounded Generation (rag.py)
   - Groq llama-3.3-70b-versatile
   - numbered context passages
   - refusal when context is missing
   - deterministic source line
     |
     v
6. Interface and Evaluation
   - Gradio chat UI in app.py
   - five-question evaluation in evaluate.py
```

## AI Tool Plan

Milestone 3 - ingestion and chunking:

I will give the AI tool the UCSD document list and chunking strategy, then ask it to
implement file ingestion and sentence-aware chunking with source metadata. I will
verify by printing chunk counts and sample chunks and checking that course names stay
near their advice.

Milestone 4 - retrieval:

I will give the AI tool the retrieval approach (`all-MiniLM-L6-v2`, ChromaDB,
top-k=3, distance threshold 0.55) and ask for a `retrieve()` function returning text,
source, and distance. I will verify by running the five evaluation questions before
generation and checking that the relevant DSC source appears.

Milestone 5 - generation, UI, and evaluation:

I will give the AI tool the grounded-generation requirements from the rubric and ask
for `generate()`, `answer()`, `app.py`, and `evaluate.py`. I will review outputs
against the expected answers, add deterministic source-line replacement if citations
are inconsistent, and keep at least one honest failure case in the README.
