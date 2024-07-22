import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import uuid


class WebScraper:
    def __init__(self, start_url, max_pages=50, max_workers=5):
        self.start_url = start_url
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.visited_urls = set()
        self.data = {"dataset_version": "1.0", "documents": []}
        self.domain = urlparse(start_url).netloc
        self.to_visit = [start_url]

    def scrape(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while self.to_visit and len(self.visited_urls) < self.max_pages:
                futures = []
                for _ in range(min(self.max_workers, len(self.to_visit))):
                    if self.to_visit:
                        url = self.to_visit.pop(0)
                        if url not in self.visited_urls:
                            futures.append(executor.submit(self._scrape_page, url))

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        self.data["documents"].append(result)

    def _scrape_page(self, url):
        if url in self.visited_urls:
            return None

        print(f"Scraping: {url}")
        self.visited_urls.add(url)

        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            if self._is_content_page(soup, url):
                page_data = self._extract_data(soup, url)
                if page_data:
                    print(f"Extracted data from: {url}")
                    new_links = self._get_links(soup, url)
                    self.to_visit.extend([link['href'] for link in new_links if link['href'] not in self.visited_urls])
                    return page_data
            else:
                print(f"Skipping non-content page: {url}")

        except requests.RequestException as e:
            print(f"Error requesting {url}: {str(e)}")
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")

        return None

    def _is_content_page(self, soup, url):
        unwanted_patterns = [
            r'/login', r'/register', r'/sign[_-]?up', r'/account',
            r'/preferences', r'/settings', r'/special:', r'/user:',
            r'/help:', r'/file:', r'/image:', r'/media', r'upload_wizard'
        ]
        if any(re.search(pattern, url, re.I) for pattern in unwanted_patterns):
            return False

        content_indicators = [
            soup.find('div', {'id': 'mw-content-text'}),
            soup.find('div', {'class': 'mw-parser-output'}),
            soup.find('article'),
            soup.find('main'),
        ]
        return any(content_indicators)

    def _extract_data(self, soup, url):
        data = {
            "id": str(uuid.uuid4()),
            "url": url,
            "title": self._get_title(soup),
            "language": self._detect_language(soup),
            "meta_description": self._get_meta_description(soup),
            "headers": self._get_headers(soup),
            "main_content": self._get_main_content(soup),
            "sections": self._get_sections(soup),
            "extracted_date": datetime.now().strftime("%Y-%m-%d"),
            "source": self._get_source(url)
        }
        return data

    def _get_title(self, soup):
        title = soup.find('title')
        return title.text.strip() if title else ''

    def _detect_language(self, soup):
        lang = soup.find('html').get('lang')
        return lang[:2] if lang else 'en'

    def _get_meta_description(self, soup):
        meta = soup.find('meta', attrs={'name': 'description'})
        return meta['content'] if meta else ''

    def _get_headers(self, soup):
        headers = {}
        for tag in ['h1', 'h2', 'h3']:
            headers[tag] = [h.text.strip() for h in soup.find_all(tag)]
        return headers

    def _get_main_content(self, soup):
        for content_class in ['mw-parser-output', 'content', 'main', 'article']:
            content = soup.find(['div', 'main', 'article'], class_=content_class)
            if content:
                for unwanted in content(['table', 'script', 'style', 'footer']):
                    unwanted.decompose()
                return self._clean_content(content.get_text(strip=True, separator=' '))
        return self._clean_content(soup.get_text(strip=True, separator=' '))

    def _get_sections(self, soup):
        sections = []
        current_section = {"title": "", "content": ""}
        for element in soup.find_all(['h2', 'p']):
            if element.name == 'h2':
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"title": element.text.strip(), "content": ""}
            elif element.name == 'p':
                current_section["content"] += " " + element.text.strip()
        if current_section["content"]:
            sections.append(current_section)
        return sections

    def _clean_content(self, content):
        content = re.sub(r'\[edit\]', '', content)
        content = re.sub(r'\s+', ' ', content)
        return content.strip()

    def _get_source(self, url):
        parsed_url = urlparse(url)
        return parsed_url.netloc

    def _get_links(self, soup, base_url):
        links = []
        for a in soup.find_all('a', href=True):
            href = urljoin(base_url, a['href'])
            if self._should_visit(href):
                links.append({'text': a.text.strip(), 'href': href})
        return links

    def _should_visit(self, url):
        parsed_url = urlparse(url)
        return (
                parsed_url.netloc == self.domain and
                url not in self.visited_urls and
                not re.search(r'\.(jpg|jpeg|png|gif|pdf)$', parsed_url.path, re.I) and
                not any(re.search(pattern, url, re.I) for pattern in [
                    r'/login', r'/register', r'/sign[_-]?up', r'/account',
                    r'/preferences', r'/settings', r'/special:', r'/user:',
                    r'/help:', r'/file:', r'/image:', r'/media', r'upload_wizard'
                ])
        )

    def save_data(self, filename='scraped_data.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)


# Usage
if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Main_Page"  # Wikipedia main page
    scraper = WebScraper(start_url, max_pages=1000, max_workers=20)
    scraper.scrape()
    scraper.save_data()
    print(f"Scraped {len(scraper.data['documents'])} pages. Data saved to scraped_data.json")