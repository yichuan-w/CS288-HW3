"""
BFS web crawler for eecs.berkeley.edu
Crawls HTML pages starting from the main EECS page.
Stores raw HTML and extracted text.
"""

import os
import re
import json
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
from bs4 import BeautifulSoup
from tqdm import tqdm


# Match URLs under eecs.berkeley.edu
EECS_URL_PATTERN = re.compile(
    r'https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?'
)

SKIP_EXTENSIONS = {
    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp',
    '.mp3', '.mp4', '.avi', '.mov', '.wmv',
    '.zip', '.tar', '.gz', '.bz2', '.rar',
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.css', '.js', '.xml', '.json', '.rss',
    '.ico', '.woff', '.woff2', '.ttf', '.eot',
}


def is_valid_eecs_url(url):
    """Check if URL is a valid EECS page to crawl."""
    parsed = urlparse(url)

    # Must be eecs.berkeley.edu
    if not re.match(r'(?:www\d*\.)?eecs\.berkeley\.edu$', parsed.hostname or ''):
        return False

    # Skip non-HTML files
    path_lower = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return False

    # Skip fragments and query-heavy URLs
    if parsed.fragment:
        return False

    return True


def normalize_url(url):
    """Normalize URL for deduplication."""
    url, _ = urldefrag(url)
    # Remove trailing slash for consistency
    if url.endswith('/') and url.count('/') > 3:
        url = url.rstrip('/')
    return url


def extract_text_from_html(html, url):
    """Extract clean text from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, 'lxml')

    # Remove script, style, nav, footer, header elements
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'noscript', 'iframe']):
        tag.decompose()

    # Get title
    title = ''
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Get main content - try common content containers first
    main_content = None
    for selector in ['main', 'article', '#content', '.content', '#main', '.main',
                     '#page-content', '.page-content', 'div[role="main"]']:
        main_content = soup.select_one(selector)
        if main_content:
            break

    if main_content is None:
        main_content = soup.find('body') or soup

    # Extract text from tables specially
    tables_text = []
    for table in main_content.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if any(cells):
                rows.append(' | '.join(cells))
        if rows:
            tables_text.append('\n'.join(rows))

    # Get all text
    text = main_content.get_text(separator='\n', strip=True)

    # Clean up excessive whitespace
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            lines.append(line)
    text = '\n'.join(lines)

    return {
        'url': url,
        'title': title,
        'text': text,
        'tables': tables_text,
    }


def extract_links(html, base_url):
    """Extract all links from HTML page."""
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


def crawl_eecs(output_dir='data/raw_pages', max_pages=5000, delay=0.2):
    """BFS crawl of eecs.berkeley.edu."""
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
    ]

    queue = deque(seed_urls)
    visited = set()
    pages = []

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; CS288-HW3-Crawler/1.0; Berkeley educational project)',
        'Accept': 'text/html,application/xhtml+xml',
    })

    pbar = tqdm(total=max_pages, desc='Crawling EECS pages')

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        url = normalize_url(url)

        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=10, allow_redirects=True)

            # Check content type
            content_type = resp.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                continue

            if resp.status_code != 200:
                continue

            html = resp.text
            if len(html) < 100:
                continue

            # Extract page data
            page_data = extract_text_from_html(html, url)

            # Skip pages with very little text
            if len(page_data['text']) < 50:
                continue

            pages.append(page_data)
            pbar.update(1)

            # Extract and queue new links
            new_links = extract_links(html, url)
            for link in new_links:
                if link not in visited:
                    queue.append(link)

            time.sleep(delay)

        except Exception as e:
            continue

    pbar.close()

    # Save all pages
    output_file = os.path.join(output_dir, 'pages.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"Crawled {len(pages)} pages, saved to {output_file}")
    return pages


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/raw_pages')
    parser.add_argument('--max-pages', type=int, default=5000)
    parser.add_argument('--delay', type=float, default=0.2)
    args = parser.parse_args()

    crawl_eecs(args.output_dir, args.max_pages, args.delay)
