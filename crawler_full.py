"""
Complete BFS crawler for eecs.berkeley.edu - crawls until queue is empty.
Uses shorter timeouts and skips pagination-heavy paths after a limit.
"""
import os, re, json, time, requests
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque, Counter
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

SKIP_EXT = {'.pdf','.jpg','.jpeg','.png','.gif','.svg','.bmp','.mp3','.mp4',
            '.avi','.mov','.zip','.tar','.gz','.bz2','.rar','.doc','.docx',
            '.xls','.xlsx','.ppt','.pptx','.css','.js','.xml','.json','.rss',
            '.ico','.woff','.woff2','.ttf','.eot','.webp','.tiff'}

# Skip query-parameter pagination duplicates beyond a threshold
MAX_PAGES_PER_PREFIX = 200  # e.g., /news/page/1..136 -> keep all

def is_valid(url):
    p = urlparse(url)
    h = p.hostname or ''
    if not re.match(r'(?:www\d*\.)?eecs\.berkeley\.edu$', h):
        return False
    for ext in SKIP_EXT:
        if p.path.lower().endswith(ext):
            return False
    return True

def norm(url):
    url, _ = urldefrag(url)
    if url.endswith('/') and url.count('/') > 3:
        url = url.rstrip('/')
    return url

def get_prefix(url):
    """Get path prefix for pagination dedup."""
    p = urlparse(url)
    # /news/page/42 -> /news/page/
    # /category/people/page/5 -> /category/people/page/
    path = p.path
    m = re.match(r'(.*/page/)\d+', path)
    if m:
        return m.group(1)
    # Query-based pagination
    if p.query:
        return p.path + '?'
    return None

def fetch(url):
    try:
        r = requests.get(url, timeout=8, allow_redirects=True,
                         headers={'User-Agent':'Mozilla/5.0 (compatible; CS288-HW3/1.0)'})
        ct = r.headers.get('Content-Type','')
        if 'text/html' not in ct or r.status_code != 200:
            return None
        html = r.text
        if len(html) < 100:
            return None

        soup = BeautifulSoup(html, 'lxml')
        for tag in soup.find_all(['script','style','noscript','iframe']):
            tag.decompose()

        title = ''
        t = soup.find('title')
        if t:
            title = t.get_text(strip=True)

        # Get main content
        mc = None
        for sel in ['main','article','#content','.content','#main','.main']:
            mc = soup.select_one(sel)
            if mc:
                break
        if mc is None:
            mc = soup.find('body') or soup

        # Tables
        tables_text = []
        for table in mc.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td','th'])]
                if any(cells):
                    rows.append(' | '.join(cells))
            if rows:
                tables_text.append('\n'.join(rows))

        text = mc.get_text(separator='\n', strip=True)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        text = '\n'.join(lines)
        if len(text) < 50:
            return None

        # Extract links from original HTML (not decomposed)
        soup2 = BeautifulSoup(html, 'lxml')
        links = set()
        for a in soup2.find_all('a', href=True):
            href = a['href'].strip()
            if href.startswith(('#','mailto:','tel:','javascript:')):
                continue
            full = norm(urljoin(url, href))
            if is_valid(full):
                links.add(full)

        return ({'url': url, 'title': title, 'text': text, 'tables': tables_text}, links)
    except:
        return None


def crawl(output_dir='data/raw_pages', workers=20):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'pages.json')

    seeds = [
        'https://eecs.berkeley.edu',
        'https://www2.eecs.berkeley.edu',
        'https://www.eecs.berkeley.edu',
        'https://eecs.berkeley.edu/people/faculty/',
        'https://eecs.berkeley.edu/academics/',
        'https://eecs.berkeley.edu/research/',
        'https://eecs.berkeley.edu/about/',
        'https://www2.eecs.berkeley.edu/Courses/',
        'https://www2.eecs.berkeley.edu/Courses/CS/',
        'https://www2.eecs.berkeley.edu/Courses/EE/',
        'https://www2.eecs.berkeley.edu/Faculty/',
        'https://www2.eecs.berkeley.edu/Pubs/Dissertations/',
        'https://www2.eecs.berkeley.edu/Research/Areas/',
        'https://www2.eecs.berkeley.edu/Scheduling/CS/schedule-draft.html',
        'https://www2.eecs.berkeley.edu/Scheduling/EE/schedule-draft.html',
        'https://eecs.berkeley.edu/people/',
        'https://eecs.berkeley.edu/resources/',
        'https://eecs.berkeley.edu/academics/courses/',
        'https://eecs.berkeley.edu/academics/graduate/',
        'https://eecs.berkeley.edu/academics/undergraduate/',
        'https://eecs.berkeley.edu/book/',
        'https://eecs.berkeley.edu/people/faculty/in-memoriam/',
        'https://eecs.berkeley.edu/people/students-2/awards/',
        'https://eecs.berkeley.edu/book/faculty/',
    ]

    visited = set()
    queue = deque()
    for u in seeds:
        u = norm(u)
        if u not in visited:
            queue.append(u)
            visited.add(u)

    pages = []
    prefix_counts = Counter()
    start = time.time()
    errors = 0

    while queue:
        # Take a batch
        batch = []
        while queue and len(batch) < workers * 2:
            url = queue.popleft()
            # Check pagination limit
            prefix = get_prefix(url)
            if prefix:
                prefix_counts[prefix] += 1
                if prefix_counts[prefix] > MAX_PAGES_PER_PREFIX:
                    continue
            batch.append(url)

        if not batch:
            break

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch, u): u for u in batch}
            for f in as_completed(futs):
                try:
                    res = f.result()
                except:
                    errors += 1
                    continue
                if res is None:
                    continue
                pd, links = res
                pages.append(pd)
                for l in links:
                    if l not in visited:
                        visited.add(l)
                        queue.append(l)

        elapsed = time.time() - start
        if len(pages) % 500 == 0 and len(pages) > 0:
            rate = len(pages) / elapsed
            print(f'Pages: {len(pages)}, Queue: {len(queue)}, '
                  f'Visited: {len(visited)}, Rate: {rate:.1f}/s, '
                  f'Errors: {errors}', flush=True)

        # Periodic save every 2000 pages
        if len(pages) % 2000 == 0 and len(pages) > 0:
            with open(output_file, 'w') as f:
                json.dump(pages, f, ensure_ascii=False)
            print(f'  Checkpoint saved ({len(pages)} pages)', flush=True)

    # Final save
    with open(output_file, 'w') as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    total_text = sum(len(p['text']) for p in pages)
    print(f'\nDONE: {len(pages)} pages in {elapsed:.1f}s')
    print(f'Visited: {len(visited)} URLs, Queue empty: {len(queue)==0}')
    print(f'Total text: {total_text:,} chars')
    print(f'Errors: {errors}')
    print(f'Saved to {output_file}')


if __name__ == '__main__':
    crawl()
