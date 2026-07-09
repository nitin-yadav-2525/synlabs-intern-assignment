# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines a retriever with a generative language model.
Instead of relying only on the parameters of the LLM, the system first retrieves
relevant chunks of text from an external knowledge base, then passes those chunks
to the LLM as context so it can produce a grounded answer.

## Why RAG is used

RAG reduces hallucination because the model is asked to answer using retrieved
evidence rather than only its internal parametric memory. It also allows the
knowledge base to be updated without retraining the model, which is much
cheaper than fine-tuning.

## Chunking

Documents are split into chunks before embedding, because embedding an entire
long document into a single vector loses fine-grained detail. Typical chunk
sizes range from 300 to 1000 characters, with an overlap of 10-20% of the
chunk size so that context is not lost at chunk boundaries.

## Vector stores

A vector store indexes embeddings so that nearest-neighbor search can be
performed efficiently. Popular options include managed services (Pinecone,
Weaviate Cloud) and self-hosted or embedded options (pgvector, Qdrant,
ChromaDB, LanceDB, FAISS, sqlite-vec). Managed services typically bill for
always-on compute pods regardless of query volume, which becomes expensive
for large but lightly-queried indexes.

## Evaluation

RAG systems are evaluated on two layers: retrieval quality (did we fetch the
right chunks?) and answer quality (did the LLM produce a correct, grounded
answer using those chunks?). Common retrieval metrics include Recall@k, Mean
Reciprocal Rank (MRR), and normalized Discounted Cumulative Gain (nDCG@k).
Common answer-quality metrics include faithfulness (is the answer supported
by the retrieved context?) and answer relevance (does the answer address the
question?).
