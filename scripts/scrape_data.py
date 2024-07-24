import argparse
import time
from kc_llm.scraping import WebScraper

def main():
    parser = argparse.ArgumentParser(description="Scrape Wikipedia data")
    parser.add_argument("--start_url", type=str, default="https://en.wikipedia.org/wiki/Main_Page", help="Starting URL for scraping")
    parser.add_argument("--max_pages", type=int, default=5000, help="Maximum number of pages to scrape")
    parser.add_argument("--output_file", type=str, default="fast_extraction_training_data.json", help="Output file for scraped data")
    args = parser.parse_args()

    start_time = time.time()
    scraper = WebScraper(args.start_url, max_pages=args.max_pages, max_workers=20, timeout=30)
    scraper.scrape()
    scraper.save_data(args.output_file)
    end_time = time.time()

    print(f"Scraped {len(scraper.data['documents'])} pages in {end_time - start_time:.2f} seconds.")
    print(f"Data saved to {args.output_file}")
    print(f"Total blacklisted patterns: {len(scraper.blacklisted_patterns)}")

if __name__ == "__main__":
    main()