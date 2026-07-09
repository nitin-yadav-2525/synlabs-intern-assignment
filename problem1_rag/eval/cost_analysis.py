"""
Run with:  python -m eval.cost_analysis
Produces a monthly cost table comparing our store (Chroma, embedded, on a
small VM/disk) against a managed vector DB, at 100K / 1M / 10M vectors.

ASSUMPTIONS (state these; change the constants below to match your own
provider quotes before submitting):
  - Embedding dimensionality: 384 floats -> 384*4 bytes = 1536 bytes/vector
    raw, but on-disk with Chroma's index overhead we assume ~2x raw size.
  - A small persistent VM/disk (e.g. a $10-20/mo cloud VM or a fraction of
    an existing app server) can hold up to ~10M such vectors on a few GB of
    SSD; we bill only the marginal disk cost since compute is already
    running the API server anyway (that's the "credible alternative" bet).
  - Managed vector DB: priced per always-on pod, roughly $70-100/month per
    pod, where one pod comfortably holds up to ~1M vectors of this
    dimensionality before you must add another pod (this mirrors typical
    published pricing tiers as of early 2026 - re-check current pricing
    pages before you rely on this for a real budget).
  - Query volume is NOT the driver of managed cost in this comparison
    (that's exactly the "lightly-queried index" problem statement) - pod
    count is.
"""
from __future__ import annotations
import json
from pathlib import Path

BYTES_PER_VECTOR = 384 * 4          # float32, 384-dim (all-MiniLM-L6-v2)
CHROMA_OVERHEAD_MULTIPLIER = 2.0     # HNSW index + metadata overhead, rough
DISK_COST_PER_GB_MONTH = 0.10        # typical SSD block storage, USD
BASE_VM_MONTHLY = 15.0               # small VM/disk already running the API

MANAGED_POD_COST_MONTHLY = 80.0
MANAGED_VECTORS_PER_POD = 1_000_000

SCALES = [100_000, 1_000_000, 10_000_000]


def chroma_monthly_cost(n_vectors: int) -> float:
    bytes_total = n_vectors * BYTES_PER_VECTOR * CHROMA_OVERHEAD_MULTIPLIER
    gb = bytes_total / (1024 ** 3)
    disk_cost = gb * DISK_COST_PER_GB_MONTH
    return round(BASE_VM_MONTHLY + disk_cost, 2)


def managed_monthly_cost(n_vectors: int) -> float:
    pods = max(1, -(-n_vectors // MANAGED_VECTORS_PER_POD))  # ceil division
    return round(pods * MANAGED_POD_COST_MONTHLY, 2)


def run():
    rows = []
    for n in SCALES:
        chroma_cost = chroma_monthly_cost(n)
        managed_cost = managed_monthly_cost(n)
        rows.append({
            "vectors": n,
            "chroma_monthly_usd": chroma_cost,
            "managed_monthly_usd": managed_cost,
            "savings_usd": round(managed_cost - chroma_cost, 2),
            "savings_pct": round((managed_cost - chroma_cost) / managed_cost * 100, 1),
        })

    out = {
        "assumptions": {
            "bytes_per_vector_raw": BYTES_PER_VECTOR,
            "chroma_overhead_multiplier": CHROMA_OVERHEAD_MULTIPLIER,
            "disk_cost_per_gb_month_usd": DISK_COST_PER_GB_MONTH,
            "base_vm_monthly_usd": BASE_VM_MONTHLY,
            "managed_pod_cost_monthly_usd": MANAGED_POD_COST_MONTHLY,
            "managed_vectors_per_pod": MANAGED_VECTORS_PER_POD,
        },
        "table": rows,
    }
    out_path = Path(__file__).parent / "results" / "cost_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    run()
