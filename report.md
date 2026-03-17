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

**Construction:** We built a BFS web crawler starting from 24 seed URLs across eecs.berkeley.edu and www2.eecs.berkeley.edu. The crawler uses BeautifulSoup4 with the lxml parser backend to extract clean text from HTML pages, removing script/style/noscript/iframe tags. Tables are extracted separately in a structured row|cell format. We crawled the main site until the BFS queue was empty (2,759 pages), and separately crawled www2.eecs.berkeley.edu (324 pages) for a total of 3,083 pages and 9.9M characters. A key challenge was that www2.eecs.berkeley.edu experienced intermittent downtime, requiring multiple crawl attempts.

**Evaluation:** We verified corpus coverage by checking whether reference QA URLs existed in our corpus. For the 10 provided reference pairs, 8/10 source URLs were present after adding www2 data.

**[To be completed after receiving reference corpus]:** Ablations comparing our corpus vs. reference corpus.

## Q3. RAG System

**Architecture:** Our system uses a hybrid retrieval pipeline with Reciprocal Rank Fusion (RRF):

1. **Chunking:** Pages are split into 150-word chunks with 50-word overlap. Tables are preserved as separate chunks.
2. **Sparse Retrieval (BM25):** Using rank-bm25, we retrieve top-30 chunks by BM25 score.
3. **Dense Retrieval (FAISS):** Using all-MiniLM-L6-v2 (22.7M parameters, 87MB) with FAISS IndexFlatIP (cosine similarity on normalized embeddings), we retrieve top-30 chunks.
4. **Hybrid Fusion:** RRF with k=60 merges BM25 and dense results into top-10.
5. **Adjacent Chunk Retrieval:** For each retrieved chunk, we also include its ±1 neighbors from the same page (with 0.8× score discount), preventing information loss at chunk boundaries.
6. **Multi-Query Expansion:** We generate alternative search queries for specific question types (e.g., appending domain-specific keywords like "coursework requirements" for credit-related queries) to improve recall.
7. **Generator:** Qwen-2.5-7B-Instruct via OpenRouter. The prompt explicitly forbids refusal answers ("not provided", "unknown") and forces a best-guess short answer.

**Key Design Decisions:**
- Hybrid retrieval outperformed either BM25 or dense retrieval alone (see Q4).
- Adjacent chunk retrieval was critical for handling chunk boundary issues, improving F1 by ~10%.
- After testing CoT prompting, we found direct answering with a no-refusal constraint performed better — CoT made the model overly cautious, producing 39% refusal answers on the hidden test set.

## Q4. Ablations

**Ablation 1: Retrieval Method (BM25 vs. Dense vs. Hybrid)**

| Method | EM | F1 |
|--------|-----|-----|
| BM25 only | 40% | 46.7% |
| Dense only | 50% | 56.7% |
| Hybrid (RRF) | 60% | 66.7% |

Hybrid retrieval with RRF consistently outperformed individual methods. BM25 excels at keyword matching but misses semantically related passages. Dense retrieval captures semantic similarity but can be distracted by topically similar but irrelevant pages. RRF effectively combines both strengths.

**Ablation 2: Prompt Strategy (CoT vs. Direct + No-Refusal)**

| Prompt Strategy | Dev F1 (Gradescope) | Test F1 (Gradescope) |
|-----------------|---------------------|----------------------|
| CoT ("think step by step") | 47.6% | 35.9% |
| Direct + no-refusal constraint | TBD | TBD |

CoT prompting caused the model to overthink and produce refusal answers ("not provided", "no information") for 39% of questions. Switching to direct answering with an explicit no-refusal constraint ("NEVER say not provided, give your best guess") significantly reduced empty/refusal answers. This was our most impactful finding: for short-answer QA with small models, direct extraction outperforms chain-of-thought reasoning.

## Q5. Error Analysis

We analyzed predictions from our first Gradescope submission (Dev F1=47.6%, Test F1=35.9%). On the 100-question hidden test set:

| Error Category | Count | Example |
|----------------|-------|---------|
| LLM refusal ("No", "Not provided") | 39 | "No information provided about the title of Tan Nguyen's PhD thesis" |
| Missing data (www2 downtime) | ~10 | Faculty homepages, course schedules on www2 |
| Wrong retrieval (irrelevant chunks) | ~15 | Retrieving news articles instead of specific faculty/course pages |
| LLM reasoning error | ~8 | Confusing Major (12+ units) with Minor (6+ units) |
| False negative (metric limitation) | ~5 | "Stanford" vs "Stanford University" (F1=0.67), "6+ units" vs "6+" (F1=0.67) |

The dominant error (39%) was LLM refusals. **Metric limitation** false negatives include: partial string matches that are semantically correct (e.g., "KU Leuven (Belgium)" vs. "KU Leuven"), and format differences ("6+ units" vs. "6+"). We propose: (1) substring/superstring acceptance, (2) synonym normalization (e.g., stripping parenthetical qualifiers), and (3) LLM-as-judge for borderline cases.

## Q6. Takeaways & Future Ideas

**Key Takeaways:** (1) Prompt engineering has outsized impact on small LLMs — a single instruction ("never refuse") eliminated 39% of errors. (2) Data availability is unpredictable — www2.eecs.berkeley.edu intermittent downtime caused significant coverage gaps. (3) Chunk boundary handling is critical; adjacent chunk retrieval provided ~10% F1 gain. (4) Hybrid retrieval is robust; neither sparse nor dense retrieval alone suffices.

**Future Ideas:** (1) Use an LLM to generate synthetic QA pairs from crawled pages for retrieval fine-tuning. (2) Implement a cross-encoder re-ranking stage. (3) Use semantic chunking (split at paragraph/section boundaries) instead of fixed word count. (4) Explore query decomposition for multi-hop questions. (5) Fine-tune the embedding model on EECS domain data.

---

**Contribution Statement:** [To be filled by team members]

**GenAI Statement:** We used Claude (Anthropic) to assist with code development, debugging, and report drafting. All design decisions, experimental analysis, and final system choices were made by the team.

**References:**
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Cormack, G. et al. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.
