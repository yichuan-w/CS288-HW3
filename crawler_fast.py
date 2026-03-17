"""
Fast concurrent BFS web crawler for eecs.berkeley.edu
Uses ThreadPoolExecutor for parallel fetching.
"""

import os
import re
import json
import time
import requests
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys


SKIP_EXTENSIONS = {
    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp',
    '.mp3', '.mp4', '.avi', '.mov', '.wmv',
    '.zip', '.tar', '.gz', '.bz2', '.rar',
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.css', '.js', '.xml', '.json', '.rss',
    '.ico', '.woff', '.woff2', '.ttf', '.eot',
}


def is_valid_eecs_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    if not re.match(r'(?:www\d*\.)?eecs\.berkeley\.edu$', hostname):
        return False
    path_lower = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return False
    return True


def normalize_url(url):
    url, _ = urldefrag(url)
    if url.endswith('/') and url.count('/') > 3:
        url = url.rstrip('/')
    return url


def extract_text_from_html(html, url):
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup.find_all(['script', 'style', 'nav', 'noscript', 'iframe']):
        tag.decompose()

    title = ''
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text(strip=True)

    main_content = None
    for selector in ['main', 'article', '#content', '.content', '#main', '.main',
                     '#page-content', '.page-content', 'div[role="main"]']:
        main_content = soup.select_one(selector)
        if main_content:
            break
    if main_content is None:
        main_content = soup.find('body') or soup

    tables_text = []
    for table in main_content.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if any(cells):
                rows.append(' | '.join(cells))
        if rows:
            tables_text.append('\n'.join(rows))

    text = main_content.get_text(separator='\n', strip=True)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)

    return {
        'url': url,
        'title': title,
        'text': text,
        'tables': tables_text,
    }


def extract_links(html, base_url):
    soup = BeautifulSoup(html, 'lxml')
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        if href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
            continue
        full_url = urljoin(base_url, href)
        full_url = normalize_url(full_url)
        if is_valid_eecs_url(full_url):
            links.add(full_url)
    return links


def fetch_url(session, url):
    """Fetch a single URL. Returns (url, html, links) or None."""
    try:
        resp = session.get(url, timeout=10, allow_redirects=True)
        content_type = resp.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            return None
        if resp.status_code != 200:
            return None
        html = resp.text
        if len(html) < 100:
            return None
        page_data = extract_text_from_html(html, url)
        if len(page_data['text']) < 50:
            return None
        new_links = extract_links(html, url)
        return (url, page_data, new_links)
    except Exception:
        return None


def crawl_eecs(output_dir='data/raw_pages', max_pages=3000, workers=20):
    os.makedirs(output_dir, exist_ok=True)

    seed_urls = [
        'https://eecs.berkeley.edu',
        'https://www2.eecs.berkeley.edu',
        'https://www.eecs.berkeley.edu',
        'https://eecs.berkeley.edu/people/faculty/',
        'https://eecs.berkeley.edu/academics/',
        'https://eecs.berkeley.edu/research/',
        'https://eecs.berkeley.edu/about/',
        'https://www2.eecs.berkeley.edu/Courses/',
        'https://www2.eecs.berkeley.edu/Faculty/',
        'https://www2.eecs.berkeley.edu/Pubs/Dissertations/',
        'https://eecs.berkeley.edu/people/',
        'https://eecs.berkeley.edu/resources/',
        'https://www2.eecs.berkeley.edu/Scheduling/',
        'https://eecs.berkeley.edu/academics/courses/',
        'https://eecs.berkeley.edu/academics/graduate/',
        'https://eecs.berkeley.edu/academics/undergraduate/',
    ]

    visited = set()
    queued = set(seed_urls)
    queue = deque(seed_urls)
    pages = []

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; CS288-HW3-Crawler/1.0; Berkeley educational project)',
        'Accept': 'text/html,application/xhtml+xml',
    })

    print(f"Starting crawl with {workers} workers, max {max_pages} pages")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while (queue or executor._work_queue.qsize() > 0) and len(pages) < max_pages:
            # Submit a batch of URLs
            futures = {}
            batch_size = min(workers * 2, max_pages - len(pages), len(queue))
            for _ in range(batch_size):
                if not queue:
                    break
                url = queue.popleft()
                url = normalize_url(url)
                if url in visited:
                    continue
                visited.add(url)
                future = executor.submit(fetch_url, session, url)
                futures[future] = url

            if not futures:
                break

            # Process completed futures
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                url, page_data, new_links = result
                pages.append(page_data)

                # Add new links to queue
                for link in new_links:
                    if link not in visited and link not in queued:
                        queued.add(link)
                        queue.append(link)

                if len(pages) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = len(pages) / elapsed
                    print(f"  Crawled {len(pages)} pages ({rate:.1f} pages/sec), "
                          f"queue: {len(queue)}, visited: {len(visited)}")

                if len(pages) >= max_pages:
                    break

    elapsed = time.time() - start_time
    print(f"\nCrawl complete: {len(pages)} pages in {elapsed:.1f}s "
          f"({len(pages)/elapsed:.1f} pages/sec)")

    # Save
    output_file = os.path.join(output_dir, 'pages.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_file}")

    # Print some stats
    total_text_len = sum(len(p['text']) for p in pages)
    urls = set(p['url'] for p in pages)
    print(f"Total text: {total_text_len:,} chars")
    print(f"Unique URLs: {len(urls)}")

    return pages


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/raw_pages')
    parser.add_argument('--max-pages', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()

    crawl_eecs(args.output_dir, args.max_pages, args.workers)
