import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import uuid
from html import unescape
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import time

class WebScraper:
    def __init__(self, start_url, max_pages=50, max_workers=20, timeout=30):
        self.start_url = start_url
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.timeout = timeout
        self.visited_urls = set()
        self.data = {"dataset_version": "5.0", "documents": []}
        self.domain = urlparse(start_url).netloc
        self.to_visit = set([start_url])
        self.pattern_counter = Counter()
        self.blacklisted_patterns = set()
        self.min_pattern_length = 20
        self.max_pattern_length = 200
        self.pattern_threshold = 3
        self.max_patterns = 10000
        self.stop_words = set(stopwords.words('english'))
        self.session = requests.Session()

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    def scrape(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = set()
            pbar = tqdm(total=self.max_pages, desc="Pages scraped")
            
            while self.to_visit and len(self.visited_urls) < self.max_pages:
                while len(futures) < self.max_workers and self.to_visit:
                    url = self.to_visit.pop()
                    if url not in self.visited_urls:
                        self.visited_urls.add(url)
                        futures.add(executor.submit(self._scrape_page, url))

                if not futures:
                    break

                done, futures = as_completed(futures), set()

                for future in done:
                    result = future.result()
                    if result:
                        self.data["documents"].append(result)
                        self._update_patterns(result)
                        pbar.update(1)

                if len(self.visited_urls) % 100 == 0:
                    tqdm.write(f"Pages visited: {len(self.visited_urls)}, Queue: {len(self.to_visit)}, Patterns: {len(self.pattern_counter)}")

            pbar.close()

    def _scrape_page(self, url):
        try:
            response = self.session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')  # Using lxml parser for speed

            if self._is_content_page(soup, url):
                page_data = self._extract_data(soup, url)
                if page_data:
                    self._update_links(soup, url)
                    return page_data
        except Exception as e:
            tqdm.write(f"Error scraping {url}: {str(e)}")
        return None

    def _is_content_page(self, soup, url):
        return bool(soup.select_one('#mw-content-text, .mw-parser-output, article, main'))

    def _extract_data(self, soup, url):
        title = self._clean_title(soup.title.string if soup.title else '')
        main_content = self._get_main_content(soup)

        if not main_content:
            return None

        return {
            "id": str(uuid.uuid4()),
            "url": url,
            "title": title,
            "extracted_date": datetime.now().isoformat(),
            "main_content": main_content,
            "categories": self._get_categories(soup),
        }

    def _clean_title(self, title):
        return re.sub(r'\s*-\s*Wikipedia.*$', '', re.sub(r'^Wikipedia:', '', title)).strip()

    def _get_categories(self, soup):
        return [link.string for link in soup.select('.CategoryTreeLabel')]

    def _get_main_content(self, soup):
        content_div = soup.select_one('#mw-content-text')
        if not content_div:
            return ""

        paragraphs = []
        for p in content_div.select('p'):
            text = self._clean_content(p.get_text())
            if len(text.split()) >= 20:  # Only include paragraphs with 20+ words
                paragraphs.append(text)

        return '\n\n'.join(paragraphs)

    def _clean_content(self, content):
        content = re.sub(r'\[\d+\]|\[edit\]', '', content)
        content = re.sub(r' {2,}', ' ', content)
        content = unescape(content)
        for pattern in self.blacklisted_patterns:
            content = content.replace(pattern, '')
        return content.strip()

    def _update_links(self, soup, base_url):
        for a in soup.select('a[href]'):
            href = urljoin(base_url, a['href'])
            if self._should_visit(href):
                self.to_visit.add(href)

    def _should_visit(self, url):
        parsed_url = urlparse(url)
        return (parsed_url.netloc == self.domain and
                url not in self.visited_urls and
                not re.search(r'\.(jpg|jpeg|png|gif|pdf)$', parsed_url.path, re.I))

    def _update_patterns(self, document):
        content = document['main_content']
        patterns = self._find_patterns(content)

        for pattern in patterns:
            self.pattern_counter[pattern] += 1
            if self.pattern_counter[pattern] >= self.pattern_threshold:
                self.blacklisted_patterns.add(pattern)

        if len(self.pattern_counter) > self.max_patterns:
            self.pattern_counter = Counter(dict(self.pattern_counter.most_common(self.max_patterns)))

    def _find_patterns(self, text):
        patterns = set()
        sentences = sent_tokenize(text)
        for sentence in sentences:
            words = sentence.split()
            for i in range(len(words)):
                for j in range(i + 1, min(i + 10, len(words) + 1)):
                    pattern = ' '.join(words[i:j])
                    if self.min_pattern_length <= len(pattern) <= self.max_pattern_length:
                        patterns.add(pattern)
        return patterns

    def save_data(self, filename='fast_extraction_training_data.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        tqdm.write(f"Data saved to {filename}")

# Usage
if __name__ == "__main__":
    start_time = time.time()
    start_url = "https://en.wikipedia.org/wiki/Main_Page"
    scraper = WebScraper(start_url, max_pages=5000, max_workers=100, timeout=30)
    scraper.scrape()
    scraper.save_data()
    end_time = time.time()
    print(f"Scraped {len(scraper.data['documents'])} pages in {end_time - start_time:.2f} seconds.")
    print(f"Data saved to fast_extraction_training_data.json")
    print(f"Total blacklisted patterns: {len(scraper.blacklisted_patterns)}")