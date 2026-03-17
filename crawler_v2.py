"""
BFS web crawler for eecs.berkeley.edu - v2 with periodic saves and threading.
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
    for selector in ['main', 'article', '#content', '.content', '#main', '.main']:
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


def fetch_one(url):
    """Fetch a single URL."""
    try:
        resp = requests.get(url, timeout=10, allow_redirects=True,
                           headers={'User-Agent': 'Mozilla/5.0 (compatible; CS288-HW3/1.0)'})
        ct = resp.headers.get('Content-Type', '')
        if 'text/html' not in ct or resp.status_code != 200:
            return None
        html = resp.text
        if len(html) < 100:
            return None
        page_data = extract_text_from_html(html, url)
        if len(page_data['text']) < 50:
            return None
        new_links = extract_links(html, url)
        return (page_data, new_links)
    except:
        return None


def crawl(output_dir='data/raw_pages', max_pages=3000, workers=15):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'pages.json')

    seed_urls = [
        'https://eecs.berkeley.edu',
        'https://www2.eecs.berkeley.edu',
        'https://eecs.berkeley.edu/people/faculty/',
        'https://eecs.berkeley.edu/academics/',
        'https://eecs.berkeley.edu/research/',
        'https://eecs.berkeley.edu/about/',
        'https://www2.eecs.berkeley.edu/Courses/',
        'https://www2.eecs.berkeley.edu/Faculty/',
        'https://www2.eecs.berkeley.edu/Pubs/Dissertations/',
        'https://eecs.berkeley.edu/people/',
        'https://eecs.berkeley.edu/resources/',
        'https://eecs.berkeley.edu/academics/courses/',
        'https://eecs.berkeley.edu/academics/graduate/',
        'https://eecs.berkeley.edu/academics/undergraduate/',
    ]

    visited = set()
    queue = deque()
    for u in seed_urls:
        u = normalize_url(u)
        if u not in visited:
            queue.append(u)
            visited.add(u)

    pages = []
    start_time = time.time()

    while queue and len(pages) < max_pages:
        # Take a batch from the queue
        batch = []
        while queue and len(batch) < workers * 2:
            url = queue.popleft()
            batch.append(url)

        if not batch:
            break

        # Fetch batch concurrently
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_url = {executor.submit(fetch_one, url): url for url in batch}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                except:
                    continue

                if result is None:
                    continue

                page_data, new_links = result
                pages.append(page_data)

                for link in new_links:
                    if link not in visited:
                        visited.add(link)
                        queue.append(link)

        # Progress
        elapsed = time.time() - start_time
        rate = len(pages) / elapsed if elapsed > 0 else 0
        print(f"Pages: {len(pages)}, Queue: {len(queue)}, Visited: {len(visited)}, "
              f"Rate: {rate:.1f}/s, Time: {elapsed:.0f}s", flush=True)

        # Periodic save
        if len(pages) % 200 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pages, f, ensure_ascii=False)
            print(f"  Saved checkpoint ({len(pages)} pages)", flush=True)

        if len(pages) >= max_pages:
            break

    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    total_text = sum(len(p['text']) for p in pages)
    print(f"\nDone: {len(pages)} pages in {elapsed:.1f}s ({len(pages)/elapsed:.1f}/s)")
    print(f"Total text: {total_text:,} chars")
    print(f"Saved to {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/raw_pages')
    parser.add_argument('--max-pages', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=15)
    args = parser.parse_args()
    crawl(args.output_dir, args.max_pages, args.workers)
