"""
Merge two corpora:
1. Existing crawled pages (reconstructed from chunks.json since pages.json doesn't exist)
2. Reference corpus from released_files/eecs_text_bs_rewritten.jsonl

For duplicate URLs, keep whichever has longer text.
For unique URLs, include both.
"""

import json
from collections import defaultdict


def reconstruct_pages_from_chunks(chunks_file):
    """Reconstruct pages from chunks.json by grouping chunks by URL and reassembling text."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Group chunks by URL
    url_chunks = defaultdict(list)
    url_titles = {}
    url_tables = defaultdict(list)

    for chunk in chunks:
        url = chunk['url']
        title = chunk['title']
        url_titles[url] = title

        if chunk['start_word'] == -1:
            # This is a table chunk
            url_tables[url].append(chunk['text'])
        else:
            url_chunks[url].append(chunk)

    pages = []
    for url in url_titles:
        title = url_titles[url]
        chunks_for_url = sorted(url_chunks.get(url, []), key=lambda c: c['start_word'])

        # Reassemble text from non-overlapping chunks
        # The chunks overlap by 50 words, so we need to stitch them carefully
        if chunks_for_url:
            # Start with the first chunk
            text_words = chunks_for_url[0]['text'].split()
            prev_start = chunks_for_url[0]['start_word']

            for c in chunks_for_url[1:]:
                curr_start = c['start_word']
                curr_words = c['text'].split()
                # Only add the non-overlapping part
                overlap = prev_start + len(text_words) - curr_start
                if overlap > 0 and overlap < len(curr_words):
                    text_words.extend(curr_words[overlap:])
                elif overlap <= 0:
                    text_words.extend(curr_words)
                prev_start = curr_start

            text = ' '.join(text_words)
        else:
            text = ''

        tables = url_tables.get(url, [])
        pages.append({
            'url': url,
            'text': text,
            'title': title,
            'tables': tables,
        })

    return pages


def load_reference_corpus(jsonl_file):
    """Load reference corpus from JSONL, extracting title from text."""
    pages = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            url = obj['url']
            text = obj.get('text', '').strip()

            if not text:
                continue

            # Extract title: first line if it starts with #
            lines = text.split('\n')
            if lines and lines[0].startswith('#'):
                title = lines[0].lstrip('#').strip()
            else:
                # Use URL as title
                title = url.rstrip('/').split('/')[-1].replace('-', ' ').title()

            pages.append({
                'url': url,
                'text': text,
                'title': title,
                'tables': [],
            })

    return pages


def merge_corpora(crawled_pages, ref_pages):
    """Merge two corpora. For duplicate URLs, keep the one with longer text."""
    # Index crawled pages by URL
    url_to_page = {}
    for page in crawled_pages:
        url_to_page[page['url']] = page

    crawled_count = len(url_to_page)
    ref_only = 0
    ref_wins = 0
    crawled_wins = 0

    for page in ref_pages:
        url = page['url']
        if url in url_to_page:
            existing = url_to_page[url]
            if len(page['text']) > len(existing['text']):
                # Keep reference version but preserve tables from crawled
                page['tables'] = existing.get('tables', []) + page.get('tables', [])
                url_to_page[url] = page
                ref_wins += 1
            else:
                crawled_wins += 1
        else:
            url_to_page[url] = page
            ref_only += 1

    merged = list(url_to_page.values())
    print(f"Crawled pages: {crawled_count}")
    print(f"Reference pages (non-empty): {len(ref_pages)}")
    print(f"Duplicates where reference won: {ref_wins}")
    print(f"Duplicates where crawled won: {crawled_wins}")
    print(f"Reference-only pages added: {ref_only}")
    print(f"Total merged pages: {len(merged)}")

    return merged


def main():
    chunks_file = 'data/datastore/chunks.json'
    ref_file = 'released_files/eecs_text_bs_rewritten.jsonl'
    output_file = 'data/raw_pages/pages_merged.json'

    print("Reconstructing pages from chunks...")
    crawled_pages = reconstruct_pages_from_chunks(chunks_file)
    print(f"Reconstructed {len(crawled_pages)} pages from chunks")

    print("\nLoading reference corpus...")
    ref_pages = load_reference_corpus(ref_file)
    print(f"Loaded {len(ref_pages)} non-empty reference pages")

    print("\nMerging corpora...")
    merged = merge_corpora(crawled_pages, ref_pages)

    # Save
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nSaved merged corpus to {output_file}")


if __name__ == '__main__':
    main()
