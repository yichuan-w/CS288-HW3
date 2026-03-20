"""
RAG pipeline: retrieval + generation for QA over EECS Berkeley data.
"""

import os
import json
import pickle
import re
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
                 top_k_final=12):
        if model_name is None:
            self.model_name = os.environ.get('LLM_MODEL', 'qwen/qwen3-8b')
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

        config_file = os.path.join(self.datastore_dir, 'config.json')
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        chunks_file = os.path.join(self.datastore_dir, 'chunks.json')
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        bm25_file = os.path.join(self.datastore_dir, 'bm25.pkl')
        with open(bm25_file, 'rb') as f:
            self.bm25 = pickle.load(f)

        faiss_file = os.path.join(self.datastore_dir, 'faiss_index.bin')
        self.faiss_index = faiss.read_index(faiss_file)

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

        k = 60  # RRF constant
        scores = {}

        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)

        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _extract_key_terms(self, question):
        """Extract key terms from the question (remove stop words)."""
        stop_words = {
            'what', 'which', 'who', 'whom', 'where', 'when', 'how', 'is', 'are',
            'was', 'were', 'do', 'does', 'did', 'the', 'a', 'an', 'in', 'of',
            'for', 'to', 'at', 'on', 'by', 'from', 'with', 'about', 'that',
            'this', 'it', 'its', 'can', 'could', 'will', 'would', 'should',
            'may', 'might', 'much', 'many', 'long', 'if', 'has', 'have', 'had',
            'be', 'been', 'being', 'and', 'or', 'not', 'no', 'you', 'your',
            'i', 'me', 'my', 'we', 'our', 'they', 'their', 'there', 'get',
        }
        words = re.findall(r'\b\w+\b', question)
        return [w for w in words if w.lower() not in stop_words]

    def generate_answer(self, question, context_chunks):
        """Generate answer using LLM with retrieved context."""
        # Build context string
        context_parts = []
        total_len = 0
        max_context_len = 8000
        for i, (idx, score) in enumerate(context_chunks):
            chunk = self.chunks[idx]
            part = f"[{i+1}] {chunk['text']}"
            if total_len + len(part) > max_context_len:
                break
            context_parts.append(part)
            total_len += len(part)

        context = '\n\n'.join(context_parts)

        prompt = f"""Answer the question about UC Berkeley EECS using the context below.

Important: You MUST give a short, concrete answer. NEVER say "not provided", "no information", "unknown", "not mentioned", or "cannot determine". Always give your best guess from the context.

Rules:
- Answer with a short span: a name, number, date, or phrase (under 10 words)
- For yes/no questions, answer only Yes or No
- For counting questions, count carefully and answer with just the number
- For questions about a specific person, find that exact person's information
- Include full identifiers (e.g., "CS 161" not "161")
- Do NOT explain your answer

Context:
{context}

Question: {question}
Answer:"""

        # Disable thinking mode for Qwen3
        if 'qwen3' in self.model_name.lower():
            prompt += " /no_think"

        try:
            answer = call_llm(prompt, model=self.model_name, max_tokens=50, temperature=0.0)
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

        # Remove common prefixes (iterate)
        changed = True
        while changed:
            changed = False
            for prefix in ['Answer:', 'The answer is:', 'The answer is',
                           'Final answer:', 'A:', 'A. ', '**Answer:**',
                           '**', 'So, ', 'Therefore, ', 'Thus, ',
                           'Based on the context, ', 'According to the context, ',
                           'Based on the provided context, ',
                           'From the context, ']:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
                    changed = True

        # Remove surrounding quotes/bold markers
        answer = answer.strip('*').strip()
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        if answer.endswith('.'):
            answer = answer[:-1].strip()

        # Detect refusals
        refusal_patterns = [
            'not provided', 'not mentioned', 'no information',
            'cannot determine', 'not specified', 'not stated',
            'not available', 'not found in', 'not clear from',
            'cannot be determined', 'insufficient information',
        ]
        answer_lower = answer.lower()
        for pattern in refusal_patterns:
            if pattern in answer_lower:
                return ""

        return answer

    def _expand_query(self, question):
        """Generate alternative search queries — generic, no hardcoded patterns."""
        expansions = [question]

        # Keyword-only version (helps BM25)
        key_terms = self._extract_key_terms(question)
        if len(key_terms) >= 2:
            expansions.append(' '.join(key_terms))

        # Strip "how many" to focus on the subject
        q_lower = question.lower()
        if 'how many' in q_lower:
            expansions.append(question.replace('How many', '').replace('how many', ''))

        return expansions

    def retrieve_multi_query(self, question, top_k=None):
        """Retrieve using multiple query expansions and merge."""
        top_k = top_k or self.top_k_final
        queries = self._expand_query(question)

        all_scores = {}
        for q in queries:
            results = self.retrieve_hybrid(q, top_k=top_k)
            for idx, score in results:
                all_scores[idx] = max(all_scores.get(idx, 0), score)

        sorted_results = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def answer_question(self, question):
        """Full RAG pipeline: retrieve + generate."""
        results = self.retrieve_multi_query(question)
        if not results:
            return ""
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

    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datastore_dir = os.path.join(script_dir, 'data', 'datastore')

    rag = RAGPipeline(datastore_dir=datastore_dir)

    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Question timed out")

    answers = []
    for i, question in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {question}")
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(45)
            answer = rag.answer_question(question)
            signal.alarm(0)
        except (TimeoutError, Exception) as e:
            print(f"  Error: {e}")
            signal.alarm(0)
            answer = ""

        answer = answer.replace('\n', ' ').strip()
        if not answer:
            answer = "unknown"
        answers.append(answer)
        print(f"  -> {answer}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for answer in answers:
            f.write(answer + '\n')

    print(f"Wrote {len(answers)} answers to {output_file}")


if __name__ == '__main__':
    main()
