# Experiment Log

## Best Config (applied to rag.py + submission zips)
- **top_k_bm25=30, top_k_dense=30, top_k_final=12**
- **No adjacent chunks**
- **max_context_len=8000 chars**
- **Qwen3-8B with /no_think**
- **27K chunks (multi-granularity), fp32 embedding model**
- Dev F1: ~56%

## Full Results

### Retrieval Parameters (ctx=6000, Qwen3-8B)
| Config | Dev EM | Dev F1 |
|--------|--------|--------|
| top_k 50/50/15, adj±2 (old default) | 41% | 48.8% |
| top_k 50/50/15, adj±1 | 42% | 49.2% |
| top_k 50/50/15, no adj | 42% | 49.7% |
| top_k 30/30/10, adj±2 | 44% | 51.8% |
| **top_k 30/30/10, no adj** | **45%** | **52.8%** |
| top_k 30/30/12, no adj | 45% | 52.8% |
| top_k 40/40/10, no adj | 45% | 52.0% |

### Context Length (top_k 30/30/10, no adj)
| Context | Dev EM | Dev F1 |
|---------|--------|--------|
| 5000 | 45% | 52.4% |
| 6000 | 45% | 52.8% |
| 7000 | 46% | 55.1% |
| **8000** | **48%** | **55.7%** |
| 9000 | 48% | 54.9% |
| 10000 | 47% | 54.0% |

### Best Config Refinement (ctx=8000, no adj)
| Config | Dev EM | Dev F1 |
|--------|--------|--------|
| top_k 30/30/10 | 48% | 55.7% |
| **top_k 30/30/12** | **49%** | **56.0%** |
| top_k 20/20/10 | 47% | 54.8% |

### LLM Models (baseline retrieval config)
| Model | Dev EM | Dev F1 |
|-------|--------|--------|
| **qwen/qwen3-8b** | **41%** | **48.8%** |
| meta-llama/llama-3-8b-instruct | 41% | 50.7% |
| meta-llama/llama-3.1-8b-instruct | 38% | 46.4% |
| allenai/olmo-3-7b-instruct | 42% | 48.4% |
| mistralai/mistral-7b-instruct | 0% | 0% (API issue) |

## Gradescope Submissions
| Sub | Config | Dev F1 | Test F1 | Total |
|-----|--------|--------|---------|-------|
| #1 | 20K fp16, dev-tuned | 44.0% | 42.6% | 42.6/60 |
| #2 | 20K fp32, generic | 44.0% | 52.5% | 49.5/60 |
| #3 | 27K fp32, generic+urls | 49.3% | 48.3% | 47.9/60 |
| #4 (ready) | 27K fp32, best config | ~56% (local) | ? | ? |

## Key Insight
Less retrieval = less noise = better performance. The model performs best with:
- Fewer but more relevant chunks (30 vs 50 candidates)
- No adjacent chunk expansion (the small 100-word chunks already provide granularity)
- More context to the LLM (8000 chars) so it sees more of the top chunks

## Additional Experiments

### RRF k Parameter (best retrieval config)
| k | Dev F1 |
|---|--------|
| 30 | 56.16% |
| 40 | 55.90% |
| 60 (default) | 55.90% |
| 80 | 56.06% |

Minimal difference — k=60 is fine.

### Retrieval Mode (best config)
| Mode | Dev F1 |
|------|--------|
| BM25 only | 44.65% |
| Dense only | 49.82% |
| **Hybrid (RRF)** | **54.56%** |

Hybrid confirmed ~10% better than either alone.

### Chunk Deduplication
| Approach | Dev F1 |
|----------|--------|
| No dedup (default) | 56.0% |
| 60% word overlap dedup | 54.4% |

Dedup hurts — removes useful overlapping context.

## Conclusion
The best config is already applied: top_k 30/30/12, no adjacent, ctx 8000, Qwen3-8B, RRF k=60.
No further changes recommended — all variations tested are either neutral or worse.
