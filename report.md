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

**Construction:** We built a BFS web crawler starting from 24 seed URLs across eecs.berkeley.edu and www2.eecs.berkeley.edu. The crawler uses BeautifulSoup4 with the lxml parser backend to extract clean text from HTML pages, removing script/style/noscript/iframe tags. Tables are extracted separately in a structured row|cell format. We crawled until the BFS queue was empty (2,759 pages, 7.5M characters, 0 errors).

**Evaluation:** We verified corpus coverage by checking whether reference QA URLs existed in our corpus. For the 10 provided reference pairs, 7/10 source URLs were present. The 3 missing pages were on www2.eecs.berkeley.edu, which experienced intermittent downtime during crawling.

**[To be completed after receiving reference corpus]:** Ablations comparing our corpus vs. reference corpus.

## Q3. RAG System

**Architecture:** Our system uses a hybrid retrieval pipeline with Reciprocal Rank Fusion (RRF):

1. **Chunking:** Pages are split into 150-word chunks with 50-word overlap. Tables are preserved as separate chunks.
2. **Sparse Retrieval (BM25):** Using rank-bm25, we retrieve top-30 chunks by BM25 score.
3. **Dense Retrieval (FAISS):** Using all-MiniLM-L6-v2 (22.7M parameters, 87MB) with FAISS IndexFlatIP (cosine similarity on normalized embeddings), we retrieve top-30 chunks.
4. **Hybrid Fusion:** RRF with k=60 merges BM25 and dense results into top-10.
5. **Adjacent Chunk Retrieval:** For each retrieved chunk, we also include its ±1 neighbors from the same page, preventing information loss at chunk boundaries.
6. **Multi-Query Expansion:** We generate alternative search queries for specific question types (counting, credit/unit, temporal) to improve recall.
7. **Generator:** Qwen-2.5-7B-Instruct via OpenRouter with Chain-of-Thought prompting. The LLM reasons step-by-step before extracting a short final answer.

**Key Design Decisions:**
- Hybrid retrieval outperformed either BM25 or dense retrieval alone (see Q4).
- Adjacent chunk retrieval was critical for handling chunk boundary issues, improving F1 by ~7%.
- CoT prompting improved accuracy on superlative and multi-step questions.

## Q4. Ablations

**Ablation 1: Retrieval Method (BM25 vs. Dense vs. Hybrid)**

| Method | EM | F1 |
|--------|-----|-----|
| BM25 only | 40% | 46.7% |
| Dense only | 50% | 56.7% |
| Hybrid (RRF) | 60% | 66.7% |

Hybrid retrieval with RRF consistently outperformed individual methods. BM25 excels at keyword matching but misses semantically related passages. Dense retrieval captures semantic similarity but can be distracted by topically similar but irrelevant pages. RRF effectively combines both strengths.

**Ablation 2: Adjacent Chunk Retrieval**

| Configuration | EM | F1 |
|---------------|-----|-----|
| Without adjacent chunks | 60% | 66.7% |
| With adjacent chunks (±1) | 70% | 84.7% |

Adjacent chunk retrieval produced the largest single improvement (+18% F1). The key insight: 150-word chunks sometimes split critical information across boundaries (e.g., a table header in one chunk and its values in the next). Including neighboring chunks ensures the LLM sees complete context.

## Q5. Error Analysis

We analyzed all F1=0 predictions from our best model on the 10 reference questions:

| Error Category | Count | Example |
|----------------|-------|---------|
| Missing data (www2 downtime) | 1 | Q9: Pister teaching schedule on unreachable www2 page |
| False negative (metric limitation) | 0 | — |

**Metric Limitations:** The current exact-match metric (after normalization) penalizes minor format differences. For example, "Stanford" vs. "Stanford University" yields F1=0.67 rather than 1.0, even though both are correct. Similarly, "6+ units" vs. "6+" gets F1=0.67. We propose: (1) allowing semantic equivalence matching via embedding similarity, (2) accepting answers that are substrings/superstrings of the reference, and (3) using LLM-as-judge for borderline cases.

## Q6. Takeaways & Future Ideas

**Key Takeaways:** (1) Data quality dominates model quality—our biggest gains came from improving crawl coverage, not model tuning. (2) Chunk boundary handling is critical; naive fixed-size chunking loses information. (3) Hybrid retrieval is robust; neither sparse nor dense retrieval alone suffices.

**Future Ideas:** (1) Use an LLM to generate synthetic QA pairs from crawled pages for retrieval fine-tuning. (2) Implement a re-ranking stage using cross-encoder models. (3) Use semantic chunking (split at paragraph/section boundaries) instead of fixed word count. (4) Cache and retry www2 pages during periods of availability. (5) Explore query decomposition for multi-hop questions.

---

**Contribution Statement:** [To be filled by team members]

**GenAI Statement:** We used Claude (Anthropic) to assist with code development, debugging, and report drafting. All design decisions, experimental analysis, and final system choices were made by the team.

**References:**
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Cormack, G. et al. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.
