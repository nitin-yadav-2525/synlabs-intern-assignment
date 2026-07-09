# Vector Databases Compared

## pgvector

pgvector is a PostgreSQL extension that adds a vector column type and
approximate nearest-neighbor indexes (IVFFlat, HNSW). It is a good choice
when a team already runs Postgres, since no new database technology needs to
be introduced. It scales to tens of millions of vectors on a single node
with proper indexing.

## Qdrant

Qdrant is a vector database written in Rust. It can run embedded, as a
single self-hosted server, or as a managed cloud service. It supports rich
metadata filtering and payload storage alongside vectors.

## ChromaDB

ChromaDB is an embedded vector database designed for simplicity. It can run
fully in-process with on-disk persistence (PersistentClient), which means no
separate server process needs to be kept running. This makes it very cheap
for small-to-medium corpora with light query volume, since the cost is just
disk space plus the compute of the process that is already running (e.g. the
API server), not a dedicated always-on database instance.

## LanceDB

LanceDB is a columnar, embedded vector database built on the Lance file
format. It is optimized for fast analytical scans in addition to vector
search and is a strong choice when you also want SQL-like filtering over
large multi-modal datasets.

## FAISS

FAISS (Facebook AI Similarity Search) is a library, not a database. It
provides extremely fast approximate nearest-neighbor search but does not
handle persistence, metadata filtering, or updates out of the box - all of
that has to be built on top of it.

## sqlite-vec

sqlite-vec is a SQLite extension for vector search. Because SQLite is a
single-file, serverless database, sqlite-vec is one of the lowest-overhead
options available, well suited to small corpora and edge/local deployments.

## Cost model of managed vector databases

Managed vector database providers commonly charge based on the number of
"pods" or compute units provisioned to hold an index in memory, and this
charge applies whether or not the index is queried. For an index that is
large (for example, hundreds of thousands to millions of vectors) but
lightly queried (a few hundred queries per day), the always-on pod cost can
dominate the total bill even though actual query compute is minimal.
