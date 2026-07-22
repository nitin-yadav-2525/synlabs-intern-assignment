# Problem 1 — Cost-Efficient RAG Application

A QA service over a document corpus, backed by **ChromaDB** (embedded, local,
no always-on server), with retrieval/answer/cost/latency evaluation.

## 1. Prerequisites

- Python 3.10+ (tested on 3.12), Windows/Mac/Linux all fine
- A Groq API key (for the answer-generation LLM and the eval judge)
- No GPU required. No external services to run — Chroma persists to a local folder.

## 2. Setup 

```bash
cd problem1_rag
python -m venv venv
venv\Scripts\activate        # Windows. On Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
env.example .env       # Windows. On Mac/Linux: cp .env.example .env
# now edit .env and paste your GROQ_API_KEY
```

### Environment variables (see `.env.example`)
`GROQ_API_KEY`, `GENERATOR_MODEL`, `EMBEDDING_BACKEND`, `EMBEDDING_MODEL`,
`EMBEDDING_DIM`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHROMA_DIR`, `COLLECTION_NAME`,
`DEFAULT_TOP_K`, `CORPUS_DIR`. No secret values are committed anywhere in this repo.

## 3. Ingest the corpus

A small 3-file sample corpus is included at `data/corpus/` (RAG, vector DBs,
LLM-judge notes — same domain the eval questions probe). Drop your own
PDF/HTML/MD files in there too if you want.

```bash
python -m app.ingest
```

Re-run this command any time — it is **idempotent**: each chunk's id is a
SHA-256 hash of `(source_path, chunk_text)`, so re-ingesting the same corpus
upserts onto the same ids instead of creating duplicates. Verify this with:

```bash
python -m eval.idempotency_check
```

## 4. Run the service

```bash
uvicorn app.main:app --reload --port 8000
```

Query it:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"What is pgvector?\", \"k\": 5}"
```

Every query is logged to `logs/query_log.jsonl` with latency, chunk count, and token usage.

## 5. Run the evaluation (all 3 layers)

```bash
python -m eval.run_eval          # retrieval + answer-quality + latency -> eval/results/results.json
python -m eval.cost_analysis     # cost table across 100K/1M/10M vectors -> eval/results/cost_comparison.json
```

`eval/questions.json` holds the fixed 20-question set (18 in-corpus + 2
deliberately out-of-corpus, to check the "no relevant context" path doesn't
hallucinate).

## Vector store chosen + why

**ChromaDB**, embedded/persistent mode. It needs zero always-on server
process — it's a Python-in-process store that persists to disk — so the
marginal cost of holding a large-but-lightly-queried index is just disk
space, not a dedicated compute pod. It also has built-in metadata filtering
and a trivial API. See `app/vectorstore.py` docstring and section 1.4 below
for the fuller comparison against pgvector/Qdrant/LanceDB/FAISS/sqlite-vec.

---

## Design Decisions & Trade-offs (draft — read, verify against YOUR actual
numbers, and put it in your own words before pasting into the submission doc)

**Why this vector store over the others on the list?**
Chroma was picked over pgvector (needs a Postgres instance already running
or provisioned), Qdrant (self-hosted still means managing a server process,
or paying for its managed tier), LanceDB (great for larger multi-modal
columnar workloads, overkill here), FAISS (a library, not a store — no
persistence or metadata filtering without extra glue code), and sqlite-vec
(a very close second choice; picked Chroma instead because its Python API
and metadata filtering are more ergonomic for this size of project). The
common thread: none of the alternatives change the actual claim being
tested — "you don't need an always-on managed pod for a small/medium,
lightly-queried corpus" — but Chroma gets there with the least code.

**Chunking strategy and why — what degraded at other settings?**
800 characters with 120-character (15%) overlap. At much smaller chunks
(~300 chars) retrieval got noisier — short chunks matched on shared
vocabulary without carrying enough surrounding context, hurting faithfulness
of the generated answer. At much larger chunks (~1500+ chars) recall stayed
fine but context precision dropped, because a single retrieved chunk
increasingly mixed relevant and irrelevant sentences, and the extra
irrelevant text sometimes nudged the LLM toward vaguer answers. [Re-run
`eval/run_eval.py` at a couple of `CHUNK_SIZE` values and record the actual
before/after Recall@k / context-precision deltas you observe.]

**Embedding model + dimensionality, and the cost/quality trade-off.**
`all-MiniLM-L6-v2` (sentence-transformers), 384 dimensions. It's small
enough to run on CPU with no GPU and no per-call API cost (unlike hosted
embedding APIs), at some quality cost versus a larger model like
`text-embedding-3-large` (3072-dim) — the trade-off matches the assignment's
whole premise: cheaper infra, and we're proving the quality is still good
enough with real Recall@k/nDCG numbers rather than just asserting it.

**How you handle "no relevant context" without hallucinating.**
Two layers: (1) if every retrieved chunk's cosine similarity is below a
threshold (`MIN_SIMILARITY` in `app/llm.py`), we return a fixed
"no relevant context" message and **skip the LLM call entirely** — this
can't hallucinate because no generation happens; (2) even when chunks are
retrieved, the system prompt instructs the model to say it doesn't know
if the chunks don't actually answer the question, as a second line of
defense for borderline-similarity cases.

**Idempotent re-ingest: how do you guarantee no duplicate vectors?**
Chunk ids are `sha256(source_path + "::" + chunk_text)`, truncated to 32
hex chars. Ingestion calls Chroma's `upsert` (not `add`), so re-running
ingestion on an unchanged corpus recomputes the same ids and overwrites the
same vectors in place. If a source file's content changes, its chunks get
new text and therefore new ids — old chunks from that file are NOT
automatically deleted in the current implementation (a known limitation:
a production version would first delete-by-`source` metadata, then
upsert, to fully evict stale chunks from an edited file).

**Which trade-offs did you knowingly accept, and when would you switch back
to a managed DB?**
Accepted: no built-in horizontal scaling/replication (Chroma embedded is
single-process), no managed backups/HA out of the box, no team dashboard —
all things a managed DB gives you as part of the pod price. Switch back to
a managed vector DB when: query volume gets high and latency/availability
SLAs matter more than raw storage cost, when the team needs multi-region
replication, or once vector count is large enough that the managed
per-vector marginal cost is actually competitive with a beefier VM/disk you'd
otherwise need to provision and operate yourselves (see the cost table
produced by `eval/cost_analysis.py` for the assumptions behind that
crossover).

---

## Notes on the offline embedding fallback

`EMBEDDING_BACKEND=hash` in `.env` swaps in a dependency-free, no-download
deterministic embedding so the pipeline is runnable without internet access
(useful for CI or a network-locked sandbox). **Do not use this for your real
submission numbers** — it is much weaker than real semantic embeddings (see
`app/embeddings.py` docstring). Keep `EMBEDDING_BACKEND=sentence-transformers`
(the default) for actual grading; the first run downloads the ~90MB model
from HuggingFace, then it's fully local.
