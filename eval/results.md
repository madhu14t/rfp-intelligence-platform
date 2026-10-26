# Retrieval evaluation (28 questions, embedder: fastembed)

| Config | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@1 | nDCG@3 | nDCG@5 | avg latency (ms) |
|---|---|---|---|---|---|---|---|---|
| vector | 0.714 | 0.929 | 0.964 | 0.817 | 0.714 | 0.837 | 0.843 | 14.7 |
| keyword | 0.714 | 0.964 | 1.0 | 0.836 | 0.714 | 0.86 | 0.866 | 1.4 |
| hybrid | 0.786 | 0.964 | 0.964 | 0.869 | 0.786 | 0.885 | 0.879 | 13.4 |
| hybrid_rerank | 0.821 | 1.0 | 1.0 | 0.911 | 0.821 | 0.925 | 0.913 | 15972.4 |

Misses (no relevant passage in top-k):
- vector: b2-09
- keyword: none
- hybrid: b2-09
- hybrid_rerank: none
