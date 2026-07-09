"""
Run with:  python -m eval.idempotency_check
Ingests the corpus twice and prints the vector count after each run. If
ingestion is truly idempotent, the count must be identical both times.
This is the evidence for submission section 1.5 "Screenshot - idempotent
re-ingest" - just screenshot this script's output.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingest import ingest_corpus
from app.vectorstore import get_collection

print("=== First ingest ===")
r1 = ingest_corpus()
print(r1)
count1 = get_collection().count()
print(f"Vector count after run 1: {count1}")

print("\n=== Second ingest (same corpus, no changes) ===")
r2 = ingest_corpus()
print(r2)
count2 = get_collection().count()
print(f"Vector count after run 2: {count2}")

print(f"\nIdempotent: {count1 == count2} (count1={count1}, count2={count2})")
