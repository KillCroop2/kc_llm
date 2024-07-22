import argparse
from kc_llm.scraping import WebScraper


def main():
    parser = argparse.ArgumentParser(description="Scrape Wikipedia data")
    parser.add_argument("--start_url", type=str, default="https://en.wikipedia.org/wiki/Main_Page", help="Starting URL for scraping")
    parser.add_argument("--max_pages", type=int, default=1000, help="Maximum number of pages to scrape")
    parser.add_argument("--output_file", type=str, default="training_data.json", help="Output file for scraped data")
    args = parser.parse_args()

    scraper = WebScraper(args.start_url, max_pages=args.max_pages)
    scraper.scrape()
    scraper.save_data(args.output_file)
    print(f"Scraped {len(scraper.data['documents'])} pages. Data saved to {args.output_file}")


if __name__ == "__main__":
    main()
