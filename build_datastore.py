"""
Build the retrieval datastore from crawled pages.
- Chunks text into passages
- Builds BM25 index
- Builds FAISS dense index using sentence-transformers
"""

import os
import json
import pickle
import re
import numpy as np
import faiss
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def chunk_text(text, url, title, chunk_size=300, overlap=50):
    """Split text into overlapping chunks of ~chunk_size words.
    Prepends page title to each chunk for better retrieval context.
    """
    words = text.split()
    if len(words) == 0:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)

        chunks.append({
            'text': chunk_text,
            'url': url,
            'title': title,
            'start_word': start,
        })

        if end >= len(words):
            break
        start = end - overlap

    return chunks


def build_datastore(pages_file='data/raw_pages/pages.json',
                    output_dir='data/datastore',
                    embedding_model_name='all-MiniLM-L6-v2',
                    chunk_size=300,
                    overlap=50):
    """Build BM25 and FAISS indexes from crawled pages."""
    os.makedirs(output_dir, exist_ok=True)

    # Load pages
    print("Loading pages...")
    with open(pages_file, 'r', encoding='utf-8') as f:
        pages = json.load(f)
    print(f"Loaded {len(pages)} pages")

    # Chunk all pages
    print("Chunking text...")
    all_chunks = []
    for page in tqdm(pages):
        # Main text chunks
        chunks = chunk_text(page['text'], page['url'], page['title'],
                           chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)

        # Also add table text as separate chunks
        for table_text in page.get('tables', []):
            if len(table_text.split()) > 10:
                all_chunks.append({
                    'text': table_text,
                    'url': page['url'],
                    'title': page['title'],
                    'start_word': -1,  # table
                })

    print(f"Created {len(all_chunks)} chunks")

    # Save chunks metadata
    chunks_file = os.path.join(output_dir, 'chunks.json')
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Build BM25 index
    print("Building BM25 index...")
    tokenized_chunks = [chunk['text'].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    bm25_file = os.path.join(output_dir, 'bm25.pkl')
    with open(bm25_file, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {bm25_file}")

    # Build dense embeddings with sentence-transformers
    print(f"Loading embedding model: {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name)

    print("Encoding chunks...")
    texts = [chunk['text'] for chunk in all_chunks]
    # Encode in batches
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=256,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype='float32')

    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim since normalized)
    index.add(embeddings)

    faiss_file = os.path.join(output_dir, 'faiss_index.bin')
    faiss.write_index(index, faiss_file)
    print(f"FAISS index saved to {faiss_file}")

    # Save model name for later loading
    config = {
        'embedding_model': embedding_model_name,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'num_chunks': len(all_chunks),
        'embedding_dim': dim,
    }
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print("Datastore build complete!")
    return all_chunks, bm25, index


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages-file', default='data/raw_pages/pages.json')
    parser.add_argument('--output-dir', default='data/datastore')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--chunk-size', type=int, default=300)
    parser.add_argument('--overlap', type=int, default=50)
    args = parser.parse_args()

    build_datastore(args.pages_file, args.output_dir, args.embedding_model,
                    args.chunk_size, args.overlap)
