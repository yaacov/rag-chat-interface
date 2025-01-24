import requests
import os
import tempfile
from urllib.parse import urlparse
from langchain_community.document_loaders import TextLoader, BSHTMLLoader
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter


def load_and_split(source: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """General function to load and split any supported file type."""
    file_type = get_file_type(source)

    # Handle URL downloads first
    local_source = download_if_url(source)

    if file_type == "html":
        return load_and_split_html(local_source, chunk_size, chunk_overlap)
    elif file_type == "markdown":
        return load_and_split_markdown(local_source, chunk_size, chunk_overlap)
    else:
        return load_and_split_text(local_source, chunk_size, chunk_overlap)


def get_file_type(source: str) -> str:
    if source.lower().endswith((".html", ".htm")):
        return "html"
    if source.lower().endswith(".md"):
        return "markdown"

    return "text"


def download_if_url(source: str) -> str:
    """If source is a URL, download it and return local path, otherwise return source."""
    if source.startswith(("http://", "https://")):
        response = requests.get(source)
        response.raise_for_status()

        # Extract original filename and extension
        parsed_url = urlparse(source)
        _, ext = os.path.splitext(parsed_url.path)
        if not ext:
            ext = ".txt"  # Default to .txt if no extension

        # Create temporary file with original extension
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"rag_temp_{os.urandom(4).hex()}{ext}")

        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        return temp_path
    return source


def load_and_split_html(source: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Handle html files specifically."""
    try:
        loader = BSHTMLLoader(source)
        documents = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load HTML file: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",  # Paragraphs
            "\n",  # Line breaks
            ". ",  # Sentences
            "? ",  # Questions
            "! ",  # Exclamations
            "; ",  # Semi-colons
            ":",  # Colons
            ",",  # Commas
            " ",  # Words
            "",  # Characters
        ],
    )
    split_documents = []
    for doc in documents:
        split_documents.extend(splitter.split_documents([doc]))

    return split_documents


def load_and_split_text(source: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Handle plain text files specifically."""
    loader = TextLoader(source)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",  # Paragraphs
            "\n",  # Line breaks
            ". ",  # Sentences
            "? ",  # Questions
            "! ",  # Exclamations
            "; ",  # Semi-colons
            ":",  # Colons
            ",",  # Commas
            " ",  # Words
            "",  # Characters
        ],
    )

    split_documents = []
    for doc in documents:
        split_documents.extend(splitter.split_documents([doc]))

    return split_documents


def load_and_split_markdown(
    source: str, chunk_size: int = 1000, chunk_overlap: int = 200
):
    """Handle markdown files specifically."""
    loader = TextLoader(source)
    documents = loader.load()

    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = []
    for doc in documents:
        split_documents.extend(splitter.split_documents([doc]))

    return split_documents
