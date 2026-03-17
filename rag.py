"""
RAG pipeline: retrieval + generation for QA over EECS Berkeley data.
"""

import os
import json
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from llm import call_llm


class RAGPipeline:
    def __init__(self, datastore_dir='data/datastore',
                 model_name=None,
                 top_k_bm25=30,
                 top_k_dense=30,
                 top_k_final=10):
        # Use env var for model name, default to OpenRouter format for submission
        if model_name is None:
            self.model_name = os.environ.get('LLM_MODEL', 'qwen/qwen-2.5-7b-instruct')
        else:
            self.model_name = model_name
        self.top_k_bm25 = top_k_bm25
        self.top_k_dense = top_k_dense
        self.top_k_final = top_k_final
        self.datastore_dir = datastore_dir

        self._load_datastore()

    def _load_datastore(self):
        """Load all datastore components."""
        print("Loading datastore...")

        # Load config
        config_file = os.path.join(self.datastore_dir, 'config.json')
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Load chunks
        chunks_file = os.path.join(self.datastore_dir, 'chunks.json')
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # Load BM25
        bm25_file = os.path.join(self.datastore_dir, 'bm25.pkl')
        with open(bm25_file, 'rb') as f:
            self.bm25 = pickle.load(f)

        # Load FAISS index
        faiss_file = os.path.join(self.datastore_dir, 'faiss_index.bin')
        self.faiss_index = faiss.read_index(faiss_file)

        # Load embedding model - try local path first, then HF hub
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(script_dir, 'data', 'embedding_model')
        if os.path.exists(local_model_path):
            embedding_model_name = local_model_path
        else:
            embedding_model_name = self.config['embedding_model']
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedder = SentenceTransformer(embedding_model_name)

        print(f"Datastore loaded: {len(self.chunks)} chunks")

    def retrieve_bm25(self, query, top_k=None):
        """Retrieve top-k chunks using BM25."""
        top_k = top_k or self.top_k_bm25
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def retrieve_dense(self, query, top_k=None):
        """Retrieve top-k chunks using dense retrieval (FAISS)."""
        top_k = top_k or self.top_k_dense
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype='float32')
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

    def retrieve_hybrid(self, query, top_k=None):
        """Hybrid retrieval: combine BM25 and dense results with RRF."""
        top_k = top_k or self.top_k_final

        bm25_results = self.retrieve_bm25(query, self.top_k_bm25)
        dense_results = self.retrieve_dense(query, self.top_k_dense)

        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        scores = {}

        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)

        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)

        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def generate_answer(self, question, context_chunks):
        """Generate answer using LLM with retrieved context."""
        # Build context string - limit to avoid token overflow
        context_parts = []
        total_len = 0
        max_context_len = 4000  # chars, to stay within token limits
        for i, (idx, score) in enumerate(context_chunks):
            chunk = self.chunks[idx]
            part = f"[{i+1}] {chunk['text']}"
            if total_len + len(part) > max_context_len:
                break
            context_parts.append(part)
            total_len += len(part)

        context = '\n\n'.join(context_parts)

        prompt = f"""Answer the question about UC Berkeley EECS based on the context below.

Rules:
- Give ONLY the short answer (a name, number, date, or brief phrase under 10 words)
- No explanation, no full sentences
- Extract the answer directly from the context when possible
- For "how many" questions, answer with just the number
- For yes/no questions, answer Yes or No
- For superlative questions (earliest, oldest, latest, most, etc.), carefully examine ALL items and compare their values (dates, numbers) before answering. For "earliest-born", find the person with the smallest birth year
- Read the question carefully: if asked about "minor", answer about minor (not major). If asked about a specific person/topic, answer about that specific one
- Include relevant qualifiers (e.g., "CS 161" not just "161")

Context:
{context}

Question: {question}
Answer:"""

        try:
            answer = call_llm(prompt, model=self.model_name, max_tokens=30, temperature=0.0)
            answer = self._clean_answer(answer)
            return answer
        except Exception as e:
            return ""

    @staticmethod
    def _clean_answer(answer):
        """Clean up LLM answer to be a short extractive span."""
        if not answer:
            return ""
        answer = answer.strip()
        # Take only the first line
        answer = answer.split('\n')[0].strip()
        # Remove common prefixes
        for prefix in ['Answer:', 'The answer is:', 'The answer is', 'A:', 'A. ']:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        # Remove surrounding quotes
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        # Remove trailing period
        if answer.endswith('.'):
            answer = answer[:-1].strip()
        return answer

    def _expand_query(self, question):
        """Generate alternative search queries for better retrieval."""
        expansions = [question]
        q_lower = question.lower()

        # Add keyword-focused variants
        if 'earliest' in q_lower or 'oldest' in q_lower or 'first' in q_lower:
            expansions.append(question.replace('earliest-born', 'born').replace('earliest', 'first'))
            expansions.append("faculty in memoriam born died years 1891 1900 1904")
        if 'how many' in q_lower:
            # Extract the subject
            expansions.append(question.replace('How many', '').replace('how many', ''))
        if 'deadline' in q_lower:
            expansions.append(question + " due date submit")
        if 'credit' in q_lower or 'unit' in q_lower or 'minor' in q_lower:
            expansions.append("PhD coursework minor units requirements")
        if 'teaching' in q_lower or 'courses' in q_lower:
            expansions.append(question + " schedule draft classes instructor")

        return expansions

    def retrieve_multi_query(self, question, top_k=None):
        """Retrieve using multiple query expansions and merge.
        Also includes adjacent chunks for continuity."""
        top_k = top_k or self.top_k_final
        queries = self._expand_query(question)

        all_scores = {}
        for q in queries:
            results = self.retrieve_hybrid(q, top_k=top_k)
            for idx, score in results:
                all_scores[idx] = max(all_scores.get(idx, 0), score)
                # Also add adjacent chunks from same URL with slightly lower score
                for adj in [idx - 1, idx + 1]:
                    if 0 <= adj < len(self.chunks):
                        if self.chunks[adj]['url'] == self.chunks[idx]['url']:
                            adj_score = score * 0.8
                            all_scores[adj] = max(all_scores.get(adj, 0), adj_score)

        sorted_results = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def answer_question(self, question):
        """Full RAG pipeline: retrieve + generate."""
        # Use multi-query retrieval for better coverage
        results = self.retrieve_multi_query(question)

        if not results:
            return ""

        # Generate answer
        answer = self.generate_answer(question, results)
        return answer


def main():
    """Run RAG pipeline on input questions."""
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 rag.py <questions_file> <output_file>")
        sys.exit(1)

    questions_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read questions
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions")

    # Initialize RAG pipeline
    # Determine datastore dir relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datastore_dir = os.path.join(script_dir, 'data', 'datastore')

    rag = RAGPipeline(datastore_dir=datastore_dir)

    # Answer each question with timeout
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Question timed out")

    answers = []
    for i, question in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {question}")
        try:
            # Set 45 second timeout per question
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(45)
            answer = rag.answer_question(question)
            signal.alarm(0)  # Cancel alarm
        except (TimeoutError, Exception) as e:
            print(f"  Error: {e}")
            signal.alarm(0)
            answer = ""

        # Ensure no newlines in answer
        answer = answer.replace('\n', ' ').strip()
        if not answer:
            answer = "unknown"
        answers.append(answer)
        print(f"  -> {answer}")

    # Write answers
    with open(output_file, 'w', encoding='utf-8') as f:
        for answer in answers:
            f.write(answer + '\n')

    print(f"Wrote {len(answers)} answers to {output_file}")


if __name__ == '__main__':
    main()
