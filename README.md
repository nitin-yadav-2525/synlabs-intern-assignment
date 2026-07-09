# Applied AI / ML Engineering — Take-Home Assignment

Two independent projects, one per problem. Each folder is fully self-contained
(its own venv, requirements, .env).

```
applied-ai-assignment/
├── problem1_rag/          # Problem 1 — Cost-Efficient RAG Application
│   ├── app/                 # ingestion, embeddings, vector store, retrieval, LLM, FastAPI service
│   ├── data/corpus/         # sample corpus (3 markdown docs) — drop your own PDF/HTML/MD here too
│   ├── eval/                # fixed 20-question eval set + retrieval/answer/cost/latency harness
│   ├── chroma_db/           # local persisted vector store (created on first ingest)
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md            # full setup/run instructions + design-decision drafts
│
├── problem2_judge/        # Problem 2 — LLM-as-Judge Evaluation Pipeline
│   ├── judge/                # schema, prompts, judge core, bias mitigations, validation, aggregation
│   ├── suites/               # 20-case test suite + 6-case adversarial probe set
│   ├── run_suite.py          # one command -> suite report + bias checks + validation
│   ├── run_ab.py             # A/B comparison of two configs, declares a winner
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md            # full setup/run instructions + design-decision drafts
│
└── submission_template.docx  # (your original file — fill it in using the two READMEs above)
```

## Quickest path to a finished submission today

1. `cd problem1_rag`, follow its README (setup → ingest → run eval → cost
   analysis). This gives you every number for submission section 1.3.
2. `cd problem2_judge`, follow its README (setup → `run_suite.py` →
   `run_ab.py`). This gives you every number for submission section 2.3.
3. Open `submission_template.docx`, paste in the real numbers from the
   `results/*.json` files each script produces, and take the two required
   screenshots (`eval/idempotency_check.py` output for 1.5, and a
   `logs/judge_log.jsonl` entry + `results/position_bias.json` entry for 2.5).
4. Write the two "walk through one query" / "explain your position-bias
   check" reflection paragraphs and the "something that broke" /
   AI-disclosure sections **yourself, in your own words** — the template is
   explicit that these must not be AI-assisted, and graders read against
   exactly that.

## What's already verified working (in this environment)

- Problem 1: ingestion, chunking (idempotent SHA-256 chunk ids), embedding,
  ChromaDB upsert/retrieval, and the "no relevant context" short-circuit path
  were all executed end-to-end here (see the conversation for the
  idempotency-check output showing count unchanged across two ingests).
- Problem 2: the structured-verdict parser (including the malformed-JSON
  fallback), suite aggregation, Cohen's kappa, and the position-bias
  flip-detection logic were all unit-tested here with mocked judge
  responses.
- What could **not** be executed in this sandbox: any step that calls the
  real Groq API (answer generation, LLM-as-judge scoring) — that
  needs your own API key, run on your own machine. Do that first, since
  every number in the submission template depends on it.
