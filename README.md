# PM Playbook

**A citation-grounded AI assistant for product managers, built from 303 episodes of Lenny's Podcast.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%2016-336791)](https://www.postgresql.org/)

**Live application:** [https://pm-playbook.streamlit.app/](https://pm-playbook.streamlit.app/)

PM Playbook turns hundreds of long-form product, growth, strategy, and leadership interviews into a searchable knowledge base. Users can ask natural-language questions and receive synthesized answers grounded in transcript excerpts, with the guest, episode, speaker, timestamp, and source link shown for verification.

<img width="579" height="392" alt="PM-Playbook" src="https://github.com/user-attachments/assets/b65c3e9e-acfa-4239-ba11-5ad4093145e7" />

<img width="882" height="377" alt="image" src="https://github.com/user-attachments/assets/9d5dab4b-8589-47f6-a1d8-357a58f88ca4" />

<img width="890" height="332" alt="image" src="https://github.com/user-attachments/assets/70db76bb-4ed6-488e-8f12-e5a844f3a5ff" />

---

## Table of Contents

- [Problem](#problem)
- [What the Application Does](#what-the-application-does)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Retrieval and RAG Flow](#retrieval-and-rag-flow)
- [Evaluation](#evaluation)
- [Monitoring and Feedback](#monitoring-and-feedback)
- [Running the Project Locally](#running-the-project-locally)
- [Rebuilding the Data and Embeddings](#rebuilding-the-data-and-embeddings)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
- [Limitations](#limitations)
- [Data Attribution](#data-attribution)

---

## Problem

Lenny's Podcast contains practical advice from experienced product leaders, founders, growth experts, and operators. However, that knowledge is distributed across **303 long-form transcripts**.

A product manager with a question such as:

- How do I know whether I have product-market fit?
- How should I prioritize a product roadmap?
- How can a product team improve retention?
- What did Brian Chesky say about hiring?
- How should a startup choose its pricing strategy?

would otherwise need to remember the relevant episode, manually search transcript files, or scrub through a 60–90 minute recording.

PM Playbook solves this problem by providing a conversational RAG application that:

1. searches the podcast corpus using hybrid text and vector retrieval;
2. sends the most relevant transcript excerpts to an LLM;
3. generates a focused, source-grounded answer;
4. shows the supporting guest, episode, speaker, timestamp, excerpt, and YouTube link;
5. logs usage, latency, token counts, estimated cost, and user feedback.

The intended users are product managers, founders, designers, growth practitioners, and anyone researching product-development practices.

---

## What the Application Does

### Tactical product-management Q&A

Ask a practical question and receive a synthesized answer based only on retrieved transcript evidence.

### Guest-specific lookup

Search for advice from a named guest, such as:

```text
What did Brian Chesky say about hiring?
```

### Topic exploration

Compare guidance across episodes on topics such as pricing, retention, user research, product-market fit, prioritization, leadership, and growth.

### Verifiable citations

Each response can include expandable source details with:

- episode guest;
- episode title;
- actual speaker;
- timestamp;
- transcript excerpt;
- YouTube link.

### Feedback and monitoring

Users can mark responses as **Helpful** or **Not helpful**. The monitoring page displays conversation, feedback, token, latency, model, and cost metrics.

---

## Live Demo

Open the deployed application:

**[https://pm-playbook.streamlit.app/](https://pm-playbook.streamlit.app/)**

The public deployment uses:

- **Streamlit Community Cloud** for the application;
- **self-managed PostgreSQL 16 on a Google Cloud Compute Engine `e2-micro` VM** (Always Free tier) for conversations, feedback, and telemetry;
- committed retrieval artifacts for reliable cloud startup;
- the OpenAI Responses API for grounded answer generation.

Try one of these sample questions:

```text
How do I find product-market fit?
How should I prioritize my roadmap?
How can I improve retention?
```

---

## Architecture

```mermaid
flowchart TD
    User[User] --> UI[Streamlit Chat UI]

    UI --> RAG[PM Playbook RAG Service]
    RAG --> Hybrid[Hybrid Retrieval]

    Hybrid --> Text[Minsearch TF-IDF Search]
    Hybrid --> Vector[SentenceTransformer Vector Search]
    Text --> RRF[Reciprocal Rank Fusion]
    Vector --> RRF

    RRF --> Selection[Source Selection and Attribution Rules]
    Selection --> Prompt[Grounded Prompt]
    Prompt --> LLM[OpenAI Responses API<br/>gpt-4o-mini]
    LLM --> UI

    UI --> DB[(PostgreSQL)]
    DB --> Monitor[Streamlit Monitoring Dashboard]

    DB -. stores .-> Conversations[Questions, Answers, Sources]
    DB -. stores .-> Telemetry[Tokens, Latency, Cost, Model]
    DB -. stores .-> Feedback[Helpful / Not Helpful]
```

### Production deployment

```text
Public user
    |
    v
Streamlit Community Cloud
    |-- Streamlit chat and monitoring pages
    |-- Minsearch text index
    |-- SentenceTransformer embeddings
    |-- Hybrid RRF retrieval
    |-- OpenAI grounded generation
    |
    v
Google Cloud Compute Engine (e2-micro) / PostgreSQL 16
    |-- conversations
    |-- feedback
    |-- usage telemetry
```

### Local development

```text
Local Streamlit application
    |
    +--> OpenAI API
    |
    +--> PostgreSQL 16 in Docker Compose
```

---

## Dataset

The source corpus is the public [ChatPRD/lennys-podcast-transcripts](https://github.com/ChatPRD/lennys-podcast-transcripts) repository.

### Corpus summary

- **303 transcript files** discovered during ingestion;
- YAML frontmatter with guest, episode title, publish date, duration, keywords, and source URLs;
- Markdown transcript body;
- three supported speaker-label formats;
- **50,910 processed chunks** in the final corpus;
- approximately **13.78 MB** in `data/chunks.parquet`.

The processed Parquet artifact is committed so reviewers can run retrieval and evaluation without rebuilding the complete ingestion pipeline first.

### Chunking strategy

The ingestion pipeline uses speaker-turn-aware, sentence-boundary chunking:

1. parse YAML frontmatter and transcript Markdown;
2. validate episode metadata with Pydantic;
3. identify speaker turns and timestamps;
4. merge short consecutive turns from the same speaker;
5. remove standalone conversational backchannels;
6. split long turns at sentence boundaries;
7. enforce a maximum of 300 words per chunk;
8. detect and flag sponsor/promotional content;
9. generate deterministic episode and chunk IDs;
10. validate and export the final Parquet dataset.

Sponsor chunks remain in the processed artifact with `is_sponsor_read=True` but are excluded from the search index.

### Supported transcript formats

```text
Speaker Name (HH:MM:SS): Dialogue
Speaker Name (MM:SS): Dialogue
Speaker Name: Dialogue
(HH:MM:SS): Continuation from the previous speaker
(MM:SS): Continuation from the previous speaker
```

Bare timestamp lines inherit the most recently identified speaker.

---

## Retrieval and RAG Flow

```text
User question
    |
    v
Minsearch text retrieval --------+
                                 |
SentenceTransformer retrieval ---+--> Reciprocal Rank Fusion
                                          |
                                          v
                                Source selection and attribution
                                          |
                                          v
                                  Grounded prompt construction
                                          |
                                          v
                                OpenAI Responses API
                                          |
                                          v
                              Answer + citations + telemetry
```

### Text retrieval

The text index uses Minsearch over:

- transcript text;
- episode title;
- guest;
- speaker name.

The baseline field boosts are:

```python
{
    "text": 1.0,
    "episode_title": 2.0,
    "guest": 3.0,
    "speaker_name": 2.0,
}
```

### Vector retrieval

Vector search uses:

- `sentence-transformers/all-MiniLM-L6-v2`;
- 384-dimensional normalized embeddings;
- dot-product ranking, equivalent to cosine similarity for normalized vectors;
- NumPy memory-mapped loading for the corpus matrix.

### Hybrid retrieval

The production retriever requests the top 20 text candidates and top 20 vector candidates, then combines their ranks using Reciprocal Rank Fusion:

```text
score(document) = sum(1 / (60 + rank))
```

Hybrid retrieval is used in production because it achieved the strongest retrieval evaluation results.

### Grounded generation

The RAG layer uses `gpt-4o-mini` through the OpenAI Responses API.

The prompt instructs the model to:

- use only the supplied transcript excerpts;
- distinguish the episode guest from the actual speaker;
- avoid attributing host questions or summaries to the guest;
- ignore tangential, promotional, introductory, and closing remarks;
- cite sources using numbered source references;
- say when the provided context is insufficient rather than inventing an answer.

### Source-selection rules

To improve attribution and relevance, the application:

1. retrieves three times the requested source count;
2. treats the top-ranked result's episode as the primary episode;
3. prioritizes guest-spoken excerpts from that episode;
4. avoids padding strong primary-episode evidence with weaker tangential episodes;
5. uses other guest-spoken excerpts when needed;
6. uses host or other-speaker excerpts only as a final fallback.

---

## Evaluation

The project evaluates retrieval and answer generation separately so that a retrieval failure is not mistaken for a prompt or model failure.

### Ground-truth dataset

A canonical retrieval-evaluation dataset was generated from high-quality transcript chunks.

It contains:

- **180 questions**;
- **180 unique expected chunk IDs**;
- **180 unique questions**;
- **139 unique episodes**.

The same ground-truth file is used for every retrieval method:

```text
data/evaluation/ground-truth.json
```

### Retrieval evaluation

All retrieval approaches were evaluated with the same:

- 180-question ground-truth dataset;
- 38,760 searchable chunks;
- exact expected `chunk_id`;
- Hit Rate@5;
- MRR@5.

| Retrieval method | Hit Rate@5 | MRR@5 | Avg. latency | Top-5 hits |
|---|---:|---:|---:|---:|
| Minsearch text | 0.4167 | 0.2706 | 47.74 ms | 75 / 180 |
| Vector (`all-MiniLM-L6-v2`) | 0.4444 | 0.3124 | **29.91 ms** | 80 / 180 |
| **Hybrid RRF** | **0.5722** | **0.4075** | 89.54 ms | **103 / 180** |

Hybrid RRF was selected because it produced:

- 28 more top-five hits than text retrieval;
- 23 more top-five hits than vector retrieval;
- approximately 37% relative improvement in Hit Rate@5 over Minsearch;
- approximately 51% relative improvement in MRR@5 over Minsearch.

The average hybrid retrieval time remained below 100 ms in the evaluation run, while LLM generation dominated total response time.

Evaluation notebook:

```text
notebooks/retrieval-evaluation.ipynb
```

### LLM generation evaluation

Two prompt variants were compared using:

- `gpt-4o-mini`;
- a deterministic sample of 30 questions;
- 30 unique source chunks;
- 30 unique episodes;
- the exact expected transcript chunk as oracle context;
- a structured LLM judge.

The judge labels were:

```text
RELEVANT
PARTLY_RELEVANT
NON_RELEVANT
```

| Prompt variant | RELEVANT | PARTLY_RELEVANT | NON_RELEVANT | Relevant rate | Avg. quality score |
|---|---:|---:|---:|---:|---:|
| **Baseline grounded prompt** | **30** | 0 | 0 | **1.0000** | **1.0000** |
| Concise prompt | 29 | 1 | 0 | 0.9667 | 0.9833 |

Efficiency comparison:

| Prompt variant | Avg. input tokens | Avg. output tokens | Avg. total tokens | Avg. latency |
|---|---:|---:|---:|---:|
| Baseline grounded prompt | 488.27 | 97.97 | 586.23 | 2,241.78 ms |
| Concise prompt | **393.27** | **72.87** | **466.13** | 2,196.59 ms |

The concise prompt used approximately 20.5% fewer total tokens, but omitted an important detail in one answer. The baseline grounded prompt was therefore selected for production because it achieved **30/30 RELEVANT** answers.

Evaluation files:

```text
notebooks/rag-evaluation.ipynb
data/evaluation/rag-evaluation-results.json
```

---

## Monitoring and Feedback

Every successful response can be logged to PostgreSQL.

### Conversation telemetry

The `conversations` table stores:

- question;
- answer;
- source metadata as JSONB;
- model;
- input tokens;
- output tokens;
- total tokens;
- estimated cost;
- latency;
- optional relevance label;
- timestamp.

### User feedback

The chat interface provides:

- **Helpful** feedback, stored as `1`;
- **Not helpful** feedback, stored as `-1`.

Feedback is linked to the corresponding conversation.

### Monitoring dashboard

The Streamlit monitoring page includes summary metrics and six charts:

1. conversations over time;
2. feedback breakdown;
3. input and output token usage over time;
4. response latency over time;
5. model usage;
6. estimated cost over time.

It also shows a table of recent conversations.

Open the **Monitoring** page from the Streamlit sidebar after starting the application.

---

## Running the Project Locally

### Prerequisites

Install:

- Python 3.11 or newer;
- [uv](https://docs.astral.sh/uv/);
- Docker and Docker Compose;
- an OpenAI API key.

The application dependencies are declared in `pyproject.toml` and fully resolved in the committed `uv.lock`.

### 1. Clone the repository

Because the transcript dataset is a Git submodule, clone recursively:

```bash
git clone --recurse-submodules https://github.com/jisha-jacob/pm-playbook.git
cd pm-playbook
```

For an existing clone that does not contain the transcript files:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a local `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your-openai-api-key

POSTGRES_DB=pm_playbook
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

Do not commit `.env`.

The local PostgreSQL values above match the defaults in `docker-compose.yaml`.

The database code also supports:

```dotenv
DATABASE_URL=postgresql://user:password@host:5432/database
POSTGRES_SSLMODE=require
```

`DATABASE_URL` is primarily intended for a hosted database, such as the
self-managed PostgreSQL instance on Google Cloud Compute Engine used in
production. See [docs/gcp-postgres-deploy.md](docs/gcp-postgres-deploy.md)
for how that instance is provisioned and configured.

### 4. Start PostgreSQL

```bash
docker compose up -d postgres
```

Verify that it is healthy:

```bash
docker compose ps
```

Optional direct health check:

```bash
docker compose exec postgres \
  pg_isready -U "${POSTGRES_USER:-postgres}" \
  -d "${POSTGRES_DB:-pm_playbook}"
```

### 5. Build embeddings only when missing

The production embedding artifacts are currently committed for reliable Streamlit deployment. If they are absent or you want to regenerate them:

```bash
uv run python scripts/build_embeddings.py
```

Expected files:

```text
data/embeddings/all-MiniLM-L6-v2-text-embeddings.npy
data/embeddings/all-MiniLM-L6-v2-text-chunk-ids.json
```

### 6. Initialize the database schema

```bash
uv run python -m pm_playbook.db_prep
```

This creates the `conversations` and `feedback` tables if they do not already exist.

### 7. Start the Streamlit application

```bash
uv run streamlit run streamlit_app/Home.py
```

Open the URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

### 8. Stop the local database

```bash
docker compose down
```

To also delete the PostgreSQL development volume and all locally stored conversations:

```bash
docker compose down -v
```

---

## Rebuilding the Data and Embeddings

The committed processed artifacts allow reviewers to run the application without repeating ingestion. The full pipeline remains reproducible.

### Run a three-transcript smoke test

```bash
uv run python -m pm_playbook.ingest \
  --limit 3 \
  --output data/chunks-smoke-test.parquet \
  --fail-fast
```

### Run a twenty-transcript quality test

```bash
uv run python -m pm_playbook.ingest \
  --limit 20 \
  --output data/chunks-quality-test.parquet \
  --fail-fast
```

### Rebuild the full processed corpus

```bash
uv run python -m pm_playbook.ingest --fail-fast
```

Output:

```text
data/chunks.parquet
```

### Rebuild vector embeddings

```bash
uv run python scripts/build_embeddings.py
```

The embedding rows are validated against the aligned chunk-ID file before production search starts.

---

## Project Structure

```text
pm-playbook/
├── data/
│   ├── raw/transcripts/                    # transcript dataset Git submodule
│   ├── chunks.parquet                      # processed retrieval corpus
│   ├── embeddings/
│   │   ├── all-MiniLM-L6-v2-text-embeddings.npy
│   │   └── all-MiniLM-L6-v2-text-chunk-ids.json
│   └── evaluation/
│       ├── evaluation-sample.parquet
│       ├── ground-truth.json
│       ├── ground-truth-review.csv
│       └── rag-evaluation-results.json
├── pm_playbook/
│   ├── __init__.py
│   ├── constants.py                        # parsing and chunking rules
│   ├── models.py                           # Pydantic data models
│   ├── utils.py                            # IDs, timestamps, and text utilities
│   ├── parser.py                           # YAML and Markdown parser
│   ├── chunker.py                          # speaker-aware chunk creation
│   ├── ingest.py                           # ingestion orchestration and validation
│   ├── search.py                           # text, vector, and hybrid retrieval
│   ├── rag.py                              # grounded RAG flow
│   ├── db.py                               # PostgreSQL schema and persistence
│   └── db_prep.py                          # database initialization
├── notebooks/
│   ├── inspect.ipynb
│   ├── chunking-experiments.ipynb
│   ├── evaluation-data-generation.ipynb
│   ├── retrieval-evaluation.ipynb
│   └── rag-evaluation.ipynb
├── scripts/
│   └── build_embeddings.py
├── streamlit_app/
│   ├── Home.py                             # chat, citations, telemetry, feedback
│   └── pages/
│       └── 1_Monitoring.py                 # monitoring dashboard
├── docker-compose.yaml                     # local PostgreSQL 16 service
├── pyproject.toml
├── uv.lock
├── cli.py
├── test.py
└── README.md
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM | OpenAI `gpt-4o-mini` |
| LLM API | OpenAI Responses API |
| Text retrieval | Minsearch |
| Vector retrieval | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Hybrid retrieval | Reciprocal Rank Fusion |
| Data processing | Pandas, PyArrow, NumPy |
| Validation | Pydantic |
| Interface | Streamlit |
| Persistence | PostgreSQL 16 |
| Database access | SQLAlchemy Core |
| Local infrastructure | Docker Compose |
| Cloud app hosting | Streamlit Community Cloud |
| Cloud database | Self-managed PostgreSQL 16 on Google Cloud Compute Engine (`e2-micro`) |
| Dependency management | uv |
| Evaluation | Hit Rate, MRR, structured LLM-as-a-judge |

---

## Design Decisions and Trade-offs

### Hybrid search instead of text-only retrieval

Text retrieval performed well for exact names and terminology but struggled with conceptual paraphrases. Vector retrieval improved semantic matching but could miss exact wording. Reciprocal Rank Fusion combined the strengths of both and produced the best evaluation result.

### Local embeddings instead of a hosted vector database

The searchable corpus contains 38,760 chunks. A normalized NumPy matrix is sufficient for the project scale, avoids an additional hosted service, and achieved acceptable latency. The trade-off is that the entire vector matrix must be available to the application process.

### Grounded prompt completeness over lower token usage

The shorter prompt reduced token usage by approximately 20.5%, but omitted one required detail during evaluation. The production prompt favors complete, well-attributed answers over the modest cost reduction.

### Streamlit for both the product and monitoring interface

A multipage Streamlit application keeps the chat, feedback, and dashboard in one codebase. This is simpler than maintaining a separate frontend and analytics service.

### PostgreSQL-only Docker Compose

Docker Compose manages the local PostgreSQL dependency. A full application image was investigated but repeatedly exceeded the available GitHub Codespaces disk allocation because of large PyTorch, Transformers, and numerical-computing layers.


### Hosted application and database are separated

Streamlit Community Cloud hosts the Python application and committed retrieval artifacts. A self-managed PostgreSQL 16 instance on a Google Cloud Compute Engine `e2-micro` VM (Always Free tier) provides persistence, keeping cloud deployment simple while preserving feedback and monitoring data at no recurring cost.

This trades away what a managed service like RDS provides out of the box: there is no automatic backup, high availability, or failover, and the Postgres port is reachable from any IP since Streamlit Community Cloud has no fixed egress addresses to allowlist — security relies on TLS and `scram-sha-256` auth instead of network restriction. See [docs/gcp-postgres-deploy.md](docs/gcp-postgres-deploy.md) for the full setup and these trade-offs in detail.

## Limitations

- The transcript dataset is a static snapshot; new podcast episodes are not ingested automatically.
- Transcript quality depends on upstream auto-generated captions.
- Timestamp links are displayed as metadata but are not converted into guaranteed timestamp-accurate deep links.
- The application supports English transcripts only.
- Retrieval uses a local embedding matrix rather than a scalable vector database.
- The production application does not run an LLM judge on every response because that would add latency and cost.
- There is no user authentication or per-user conversation history.
- The ingestion pipeline is a Python script rather than an orchestrated workflow such as Airflow, Prefect, Kestra, Mage, or dlt.
- The full Streamlit application is not included in Docker Compose.
- User query rewriting is not currently implemented.
- Evaluation results are based on the selected ground-truth and judge methodology and should not be interpreted as universal performance guarantees.

---

## Data Attribution

The transcript source is:

[ChatPRD/lennys-podcast-transcripts](https://github.com/ChatPRD/lennys-podcast-transcripts)

The source repository states that the transcripts are for personal and educational use and that the underlying content belongs to Lenny's Podcast and its guests.

This project is an independent educational project created for the DataTalksClub LLM Zoomcamp. It is not affiliated with or endorsed by Lenny's Podcast, its guests, ChatPRD, Streamlit, OpenAI, or Google Cloud.

---

## Acknowledgements

- [DataTalksClub LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp)
- [Lenny's Podcast](https://www.lennyspodcast.com/)
- [ChatPRD/lennys-podcast-transcripts](https://github.com/ChatPRD/lennys-podcast-transcripts)
