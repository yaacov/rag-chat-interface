import argparse
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Set
from urllib.parse import urljoin, urlparse


def extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    return urlparse(url).netloc


def is_http_url(url: str) -> bool:
    """Check if a URL uses HTTP (non-TLS) protocol."""
    return url.startswith('http://') and not url.startswith('https://')


def should_ignore_url(url: str, ignore_patterns: List[str], ignore_http: bool = False) -> bool:
    """Check if URL matches any of the ignore patterns or if it's HTTP and should be ignored."""
    if ignore_http and is_http_url(url):
        return True
    return any(re.search(pattern, url) for pattern in ignore_patterns)


def clean_url(url: str) -> str:
    """Remove anchor fragments from URL."""
    return url.split('#')[0]


def get_links_from_page(url: str) -> List[str]:
    """Extract all unique links from a web page, clean them, and convert to absolute URLs."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        
        for link_tag in soup.find_all("a", href=True):
            href = link_tag["href"]
            
            # Skip anchor-only links
            if href.startswith('#'):
                continue
                
            # Make URL absolute and clean it
            absolute_url = urljoin(url, href)
            clean_link = clean_url(absolute_url)
            
            links.add(clean_link)
            
        return list(links)
    except requests.RequestException:
        return []


def format_display_url(url: str) -> str:
    """Format URL for display - adds index.html if needed."""
    if url.endswith('.html'):
        return url
    
    if url.endswith('/'):
        return f"{url}index.html"
    
    if '.' not in url.split('/')[-1]:
        return f"{url}/index.html"
    
    return url


def crawl_domain(start_url: str, ignore_patterns: List[str] = None, verbose: bool = False, ignore_http: bool = False) -> List[str]:
    """
    Crawls the domain of the given start_url.
    
    Args:
        start_url: The URL to start crawling from
        ignore_patterns: A list of regex patterns to ignore
        verbose: Whether to print detailed output
        ignore_http: Whether to ignore HTTP (non-TLS) links
        
    Returns:
        A list of URLs found within the same domain
    """
    if ignore_patterns is None:
        ignore_patterns = []
    
    domain = extract_domain(start_url)
    start_url = clean_url(start_url)
    
    visited_urls: Set[str] = set()
    to_visit: List[str] = [start_url]
    count = 0

    while to_visit:
        current_url = to_visit.pop()
        
        # Skip if already visited or should be ignored
        if current_url in visited_urls:
            continue
            
        if should_ignore_url(current_url, ignore_patterns, ignore_http):
            if verbose:
                if ignore_http and is_http_url(current_url):
                    print(f"Skipping (non-TLS HTTP link): {current_url}")
                else:
                    print(f"Skipping (matches ignore pattern): {current_url}")
            continue

        # Process current URL
        visited_urls.add(current_url)
        count += 1
        
        if verbose:
            print(f"[{count}] Visiting: {current_url}")
        else:
            print(".", end="", flush=True)
            if count % 50 == 0:
                print(f" {count}")

        # Get and process links from the current page
        for absolute_url in get_links_from_page(current_url):
            # Only process URLs within the same domain
            if extract_domain(absolute_url) == domain:
                if not should_ignore_url(absolute_url, ignore_patterns, ignore_http) and absolute_url not in visited_urls:
                    to_visit.append(absolute_url)

    if not verbose:
        print()
        
    return list(visited_urls)


def get_default_ignore_patterns() -> List[str]:
    """Get the default list of patterns to ignore."""
    return [
        r'/v\d+\.\d+/', # Skip versioned docs
        r'/20\d\d/',    # Skip year-based URLs
        r'/api/',       # Skip API documentation
        r'/blog/',      # Skip blog posts
        r'/tag/',       # Skip tag pages
        r'/blogs/',     # Skip blog posts
        r'/tags/',      # Skip tag pages
        r'/search/',    # Skip search pages
        r'/page/\d+/',  # Skip pagination pages
        r'\.(jpg|jpeg|png|gif|webp|svg|ico|bmp)$',  # Ignore image files
        r'\.(mp4|webm|avi|mov|wmv|flv|mkv)$',       # Ignore video files
    ]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Web crawler that finds all URLs within a domain.")
    parser.add_argument(
        "start_url", 
        help="URL to start crawling from (example: https://kubevirt.io/user-guide/)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed output during crawling"
    )
    parser.add_argument(
        "-i", "--ignore-http",
        action="store_true",
        help="Ignore HTTP (non-TLS) links during crawling"
    )
    return parser.parse_args()


def main():
    """Main function to run the crawler."""
    args = parse_arguments()
    start_url = args.start_url
    ignore_patterns = get_default_ignore_patterns()
    
    print(f"Starting crawler at: {start_url}")
    if args.verbose:
        print(f"Ignoring URLs that match these patterns: {ignore_patterns}")
    if args.verbose and args.ignore_http:
        print("Ignoring HTTP (non-TLS) links")
    
    all_urls = crawl_domain(start_url, ignore_patterns, args.verbose, args.ignore_http)
    
    print(f"\n{len(all_urls)} URLs found within the same domain:\n")
    for url in all_urls:
        print(format_display_url(url))


if __name__ == "__main__":
    main()

