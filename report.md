# CS288 Assignment 3: Building a RAG System for UC Berkeley EECS QA

## Q1. QA Data Creation

We created a QA dataset by manually browsing the EECS website and formulating factoid questions across diverse topics: faculty information, degree requirements, course details, awards, and administrative deadlines. We aimed for questions spanning different difficulty levels—from those answerable by LLM world knowledge to those requiring specific page retrieval.

**Dataset Statistics:**
- Total questions: 110
- Question types: extractive (90%), counting/arithmetic (5%), yes/no (5%)
- Source pages: 45 unique URLs across eecs.berkeley.edu

**Inter-Annotator Agreement (IAA):** Two annotators independently answered 35 questions (32% of the dataset). We computed exact-match agreement and token-level F1. Exact-match agreement was 74.3% (26/35), and average pairwise F1 was 82.1%. Most disagreements arose from answer granularity (e.g., "Stanford" vs. "Stanford University") and questions with multiple valid answers.

**Sample Questions:**

| Question | Answer | Source URL |
|----------|--------|------------|
| What is the course number for the class covering cryptography and network security? | CS 161 | eecs.berkeley.edu/Courses/CS/ |
| Who was the earliest-born professor in EECS at UC Berkeley? | Lester Edwin Reukema | eecs.berkeley.edu/people/faculty/in-memoriam |
| When is the deadline for outstanding TA award nominations? | 2/18/26 | eecs.berkeley.edu/people/students-2/awards/ |

## Q2. Retrieval Corpus

**Construction:** We built a BFS web crawler starting from 24 seed URLs across eecs.berkeley.edu and www2.eecs.berkeley.edu. The crawler uses BeautifulSoup4 to extract clean text from HTML, removing script/style tags. Tables are extracted separately. We crawled 3,083 pages total. We then merged 857 additional pages from the released reference corpus, prioritizing its cleaner text for 99 pages where our crawled data contained excessive HTML navigation boilerplate. The final corpus spans ~3,940 unique URLs.

**Ablation — Our Corpus vs. Reference Corpus:**

| Corpus | Dev F1 |
|--------|--------|
| Our crawled corpus only (3,083 pages) | ~49% |
| Reference corpus only (1,942 pages) | ~35% |
| Merged (ours + reference new URLs + quality replacement) | ~53% |

The reference corpus alone performed poorly due to lower page coverage. However, its text quality was superior — clean markdown without HTML boilerplate. The merged approach retained our broader coverage while benefiting from reference corpus quality for faculty homepages and publication pages.

## Q3. RAG System

**Architecture:** Our final system uses a hybrid retrieval pipeline with multi-granularity chunking:

1. **Multi-Granularity Chunking:** Primary chunks are 300 words with 50-word overlap. We additionally create 100-word chunks (20-word overlap) for all pages, enabling precise retrieval of specific facts (e.g., individual awards, contact info) that get diluted in larger chunks. Total: ~27K chunks.
2. **Sparse Retrieval (BM25):** Top-50 chunks by BM25 score.
3. **Dense Retrieval (FAISS):** all-MiniLM-L6-v2 (22.7M params, 87MB) with FAISS IndexFlatIP. Top-50 chunks.
4. **Hybrid Fusion:** RRF (k=60) merges BM25 and dense results into top-15.
5. **Adjacent Chunk Retrieval:** For each retrieved chunk, ±2 neighbors from the same page are included (0.7x score discount).
6. **Multi-Query Expansion:** Key terms are extracted from the question (stop-word removal) and used as an additional keyword query. For "how many" questions, the counting phrase is stripped to focus on the subject.
7. **Generator:** Qwen3-8B via OpenRouter with `/no_think` to disable chain-of-thought. The prompt forbids refusal answers and forces short extractive spans. Context window is 6000 characters.

**Key Design Decisions:**
- Hybrid retrieval outperformed either BM25 or dense retrieval alone (see Q4).
- We found that **avoiding dev-set-specific tuning** (e.g., hardcoded query expansions, question-type-specific prompt branches) was critical for generalization. An earlier version with dev-tuned query expansions scored 53% on dev but only 43% on test; simplifying to generic retrieval improved test F1 to 52.5% despite a small dev decrease.

## Q4. Ablations

**Ablation 1: Generalization vs. Dev-Set Tuning**

| Approach | Dev F1 | Test F1 |
|----------|--------|---------|
| Dev-tuned (hardcoded query expansions, question-type prompts) | 53% | 43% |
| Generic (keyword extraction only, uniform prompt) | 49% | 52.5% |

This was our most surprising finding. Dev-specific optimizations (e.g., adding "faculty in memoriam born died years 1891" as a query expansion for "earliest-born" questions) improved dev scores but severely hurt test generalization. The generic approach with simple keyword extraction outperformed on the unseen test set by nearly 10%.

**Ablation 2: LLM Model (Qwen 2.5 7B vs. Qwen3 8B)**

| Model | Dev F1 | Test F1 |
|-------|--------|---------|
| Qwen-2.5-7B-Instruct | ~49% | — |
| Qwen3-8B (`/no_think`) | ~53% | ~52% |
| Llama-3.1-8B-Instruct | ~46% | — |

Qwen3-8B with thinking disabled was the strongest model. Notably, Qwen3's default thinking mode was counterproductive — it caused slow responses and verbose reasoning that degraded short-answer extraction. The `/no_think` flag was essential for both speed and accuracy.

## Q5. Error Analysis

We analyzed 100 hidden dev set predictions from our best model (Qwen3-8B, Dev F1≈49%). We categorized errors using retrieval diagnostics — checking whether the answer string exists in retrieved chunks, in the corpus, or is entirely missing.

| Error Category | Count | Example |
|----------------|-------|---------|
| Retrieval has answer, LLM wrong | ~25 | Q: "How many Turing winners?" Pred: 6 vs. Ref: 9 — LLM undercounts |
| Retrieval miss, corpus has answer | ~18 | Q: "Dan Klein fellowship in 2007?" — chunk ranks >200th despite containing "Sloan Research Fellowship" |
| Corpus missing entirely | ~2 | Q: "Who earned a PhD in 1935?" — page not crawled |
| Metric limitation | ~4 | "brewer@cs.berkeley.edu" vs "brewer@cs" (F1=0) |

The dominant error source is the LLM extracting incorrect information from context that does contain the answer (25 cases). Retrieval misses (18 cases) occur when answer-bearing chunks rank too low — often because the answer is in a large, topic-diverse chunk where the signal is diluted.

**Metric limitations:** The token-level F1 metric penalizes format differences that are semantically equivalent. We propose: (1) substring acceptance for email/phone formats, (2) time normalization (e.g., "11 am" = "11:00am"), and (3) LLM-as-judge evaluation for borderline cases.

## Q6. Takeaways & Future Ideas

**Key Takeaways:** (1) **Avoiding overfitting to the dev set was our most important lesson** — simplifying our pipeline improved test F1 by 10% despite lowering dev scores. (2) Multi-granularity chunking improved retrieval precision for specific facts without sacrificing broad context. (3) Corpus quality matters: replacing boilerplate-heavy pages with clean reference text improved F1 by ~4%. (4) Newer models (Qwen3 vs. 2.5) provide meaningful gains (~4% F1) with no code changes. (5) Chain-of-thought hurts small models on extractive QA — direct answering with anti-refusal constraints works better.

**Future Ideas:** (1) Cross-encoder re-ranking to improve retrieval precision — this could address the 25 cases where the answer is in context but the LLM picks wrong info. (2) Semantic chunking at section/heading boundaries instead of fixed word count. (3) Two-pass generation for counting questions: first enumerate candidates, then count. (4) Fine-tune the embedding model on EECS domain QA pairs. (5) Entity-aware retrieval: extract named entities from questions and do targeted chunk lookup.

---

**Contribution Statement:** [To be filled by team members]

**GenAI Statement:** We used Claude (Anthropic) to assist with code development, debugging, and report drafting. All design decisions, experimental analysis, and final system choices were made by the team.

**References:**
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Cormack, G. et al. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.
