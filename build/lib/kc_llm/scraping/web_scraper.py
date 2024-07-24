import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime
import uuid
from html import unescape
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class WebScraper:
    def __init__(self, start_url, max_pages=50, max_workers=5, timeout=30):
        self.start_url = start_url
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.timeout = timeout
        self.visited_urls = set()
        self.data = {"dataset_version": "3.0", "documents": []}
        self.domain = urlparse(start_url).netloc
        self.to_visit = [start_url]
        self.pattern_counter = Counter()
        self.blacklisted_patterns = set()
        self.min_pattern_length = 20
        self.max_pattern_length = 200
        self.pattern_threshold = 3
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.max_patterns = 10000  # Limit the number of patterns
        self.max_links = 100000  # Limit the number of links to visit

        # Download NLTK data
        self._download_nltk_data()

        self.stop_words = set(stopwords.words('english'))

    def _download_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            

    def scrape(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            pbar = tqdm(total=self.max_pages, desc="Pages scraped")
            while self.to_visit and len(self.visited_urls) < self.max_pages:
                futures = []
                for _ in range(min(self.max_workers, len(self.to_visit))):
                    if self.to_visit:
                        url = self.to_visit.pop(0)
                        if url not in self.visited_urls:
                            futures.append(executor.submit(self._scrape_page, url))

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.timeout)
                        if result:
                            self.data["documents"].append(result)
                            self._update_patterns(result)
                            pbar.update(1)
                    except TimeoutError:
                        tqdm.write(f"Timeout occurred while scraping a page")
                    except Exception as e:
                        tqdm.write(f"An error occurred: {str(e)}")

                tqdm.write(f"Pages visited: {len(self.visited_urls)}, Pages to visit: {len(self.to_visit)}, Patterns: {len(self.pattern_counter)}")
            pbar.close()

    def _scrape_page(self, url):
        if url in self.visited_urls:
            return None

        tqdm.write(f"Scraping: {url}")
        self.visited_urls.add(url)

        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            if self._is_content_page(soup, url):
                page_data = self._extract_data(soup, url)
                if page_data:
                    tqdm.write(f"Extracted data from: {url}")
                    new_links = self._get_links(soup, url)
                    self.to_visit.extend([link['href'] for link in new_links if link['href'] not in self.visited_urls])
                    self.to_visit = list(dict.fromkeys(self.to_visit))  # Remove duplicates
                    if len(self.to_visit) > self.max_links:
                        self.to_visit = self.to_visit[:self.max_links]
                    return page_data
            else:
                tqdm.write(f"Skipping non-content page: {url}")

        except requests.RequestException as e:
            tqdm.write(f"Error requesting {url}: {str(e)}")
        except Exception as e:
            tqdm.write(f"Error scraping {url}: {str(e)}")

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
        title = self._clean_title(self._get_title(soup))
        main_content = self._get_main_content(soup)

        if not self._is_meaningful_content(main_content):
            return None

        data = {
            "id": str(uuid.uuid4()),
            "url": url,
            "title": title,
            "language": self._detect_language(soup),
            "extracted_date": datetime.now().strftime("%Y-%m-%d"),
            "categories": self._get_categories(soup),
            "main_content": main_content,
            "sections": self._get_sections(soup),
            "related_pages": self._get_related_pages(soup)
        }
        return data

    def _get_title(self, soup):
        title = soup.find('title')
        return title.text.strip() if title else ''

    def _clean_title(self, title):
        title = re.sub(r'\s*-\s*Wikipedia.*$', '', title)
        title = re.sub(r'^Wikipedia:', '', title)
        return title.strip()

    def _detect_language(self, soup):
        lang = soup.find('html').get('lang')
        return lang[:2] if lang else 'en'

    def _get_categories(self, soup):
        categories = []
        category_links = soup.find_all('a', {'class': 'CategoryTreeLabel'})
        for link in category_links:
            categories.append(link.text.strip())
        return categories

    def _get_main_content(self, soup):
        main_content = ""
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if content_div:
            for element in content_div.find_all(['p', 'h2', 'h3', 'ul', 'ol']):
                if element.name in ['h2', 'h3']:
                    main_content += f"\n\n{element.text.strip()}\n\n"
                elif element.name in ['p']:
                    paragraph_text = self._clean_content(element.text.strip())
                    if self._is_meaningful_paragraph(paragraph_text):
                        main_content += paragraph_text + "\n\n"
                elif element.name in ['ul', 'ol']:
                    list_text = self._process_list(element)
                    if list_text:
                        main_content += list_text + "\n\n"
        return main_content.strip()

    def _process_list(self, list_element):
        list_items = list_element.find_all('li', recursive=False)
        if len(list_items) > 5:  # Only process lists with more than 5 items
            return self._summarize_list(list_items)
        return ""

    def _summarize_list(self, list_items):
        item_texts = [item.text.strip() for item in list_items]
        summary = f"A list of {len(item_texts)} items, including: "
        summary += ", ".join(item_texts[:3])  # Include first 3 items
        if len(item_texts) > 3:
            summary += f", and {len(item_texts) - 3} more."
        return summary

    def _is_meaningful_paragraph(self, text):
        words = text.split()
        if len(words) < 20:  # Ignore very short paragraphs
            return False
        non_stop_words = [word for word in words if word.lower() not in self.stop_words]
        if len(non_stop_words) / len(words) < 0.5:  # Ignore paragraphs with too many stop words
            return False
        return True

    def _is_meaningful_content(self, content):
        sentences = sent_tokenize(content)
        if len(sentences) < 5:  # Ignore very short content
            return False
        words = content.split()
        if len(words) < 100:  # Ignore content with few words
            return False
        return True

    def _get_sections(self, soup):
        sections = []
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if content_div:
            current_section = {"title": "", "content": ""}
            for element in content_div.find_all(['h2', 'p', 'ul', 'ol']):
                if element.name == 'h2':
                    if current_section["content"] and self._is_meaningful_content(current_section["content"]):
                        sections.append(current_section)
                    current_section = {"title": element.text.strip(), "content": ""}
                elif element.name == 'p':
                    paragraph_text = self._clean_content(element.text.strip())
                    if self._is_meaningful_paragraph(paragraph_text):
                        current_section["content"] += paragraph_text + "\n\n"
                elif element.name in ['ul', 'ol']:
                    list_text = self._process_list(element)
                    if list_text:
                        current_section["content"] += list_text + "\n\n"
            if current_section["content"] and self._is_meaningful_content(current_section["content"]):
                sections.append(current_section)
        return sections

    def _clean_content(self, content):
        content = re.sub(r'\[\d+\]', '', content)  # Remove citation numbers
        content = re.sub(r'\[edit\]', '', content)  # Remove edit links
        content = re.sub(r'<[^>]+>', '', content)  # Remove any remaining HTML tags
        content = re.sub(r' {2,}', ' ', content)  # Replace multiple spaces with a single space
        content = unescape(content)  # Unescape HTML entities
        content = re.sub(r'\n{3,}', '\n\n', content)  # Remove redundant newlines

        for pattern in self.blacklisted_patterns:
            content = content.replace(pattern, '')

        return content.strip()

    def _get_related_pages(self, soup):
        related_pages = []
        links = soup.find_all('a', href=True)
        for link in links:
            if link['href'].startswith('/wiki/') and ':' not in link['href']:
                related_pages.append({
                    "title": link.text.strip(),
                    "url": urljoin(self.start_url, link['href'])
                })
        return related_pages[:5]

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

    def _update_patterns(self, document):
        content = document['main_content']
        patterns = self._find_patterns(content)

        for pattern in patterns:
            self.pattern_counter[pattern] += 1
            if self.pattern_counter[pattern] >= self.pattern_threshold:
                self.blacklisted_patterns.add(pattern)

        # Limit the number of patterns
        if len(self.pattern_counter) > self.max_patterns:
            self.pattern_counter = Counter(dict(self.pattern_counter.most_common(self.max_patterns)))

        tqdm.write(f"Current blacklisted patterns: {len(self.blacklisted_patterns)}")
        tqdm.write(f"Total patterns found: {len(self.pattern_counter)}")

    def _find_patterns(self, text):
        patterns = set()
        sentences = sent_tokenize(text)
        for sentence in sentences:
            words = sentence.split()
            for i in range(len(words)):
                for j in range(i + 1, min(i + 10, len(words) + 1)):  # Limit the pattern length
                    pattern = ' '.join(words[i:j])
                    if self.min_pattern_length <= len(pattern) <= self.max_pattern_length:
                        patterns.add(pattern)
        return patterns

    def save_data(self, filename='advanced_training_data.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        tqdm.write(f"Data saved to {filename}")


# Usage
if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Main_Page"
    scraper = WebScraper(start_url, max_pages=5000, max_workers=20, timeout=60)
    scraper.scrape()
    scraper.save_data()
    print(f"Scraped {len(scraper.data['documents'])} pages. Data saved to dynamic_training_data.json")
    print(f"Total blacklisted patterns: {len(scraper.blacklisted_patterns)}")