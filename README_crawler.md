# Web Crawler

A simple web crawler tool designed to discover and list all URLs within a specific domain. This tool can be useful for creating site maps, preparing URLs for content scraping, or testing site connectivity.

## Features

- Crawls all pages within the same domain as the starting URL
- Automatically converts relative links to absolute URLs
- Removes anchor fragments from URLs (e.g., `#section1`)
- Provides options for verbose logging during crawling
- Can ignore non-TLS (HTTP) links for security-conscious crawling
- Ignores common patterns like API documentation, versioned docs, image files, etc.
- Formats URLs for readability (adds index.html for directory URLs)

## Installation

The crawler requires Python 3 and the following dependencies:

```bash
pip install requests beautifulsoup4
```

## Usage

Basic usage:

```bash
python utils/crawler.py https://example.com/docs/
```

With verbose output:

```bash
python utils/crawler.py https://example.com/docs/ --verbose
```

### Arguments

- `start_url`: URL to start crawling from (required)
- `-v, --verbose`: Print detailed output during crawling
- `--ignore-http`: Ignore non-TLS (HTTP) links

## Default Ignored Patterns

The crawler automatically ignores the following URL patterns:

- Version-specific documentation (`/v1.0/`, etc.)
- Year-based URLs (`/2023/`, etc.)
- API documentation (`/api/`)
- Blog posts (`/blog/`, `/blogs/`)
- Tag and search pages (`/tag/`, `/tags/`, `/search/`)
- Pagination pages (`/page/123/`)
- Media files (images, videos, etc.)

## Examples

Crawling documentation site:

```bash
python utils/crawler.py https://kubevirt.io/user-guide/
```

Crawling with verbose output:

```bash
python utils/crawler.py https://kubernetes.io/docs/ --verbose
```

## Output

The crawler will print:
- A progress indicator (dots) during crawling (in non-verbose mode)
- A complete list of discovered URLs when finished

## Limitations

- Follows only HTML links (no JavaScript-generated links)
- Doesn't handle authentication
- Respects only same-domain boundaries (won't follow external links)
- No rate limiting (use responsibly to avoid overloading servers)
