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
