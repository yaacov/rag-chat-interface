import requests
import os
import tempfile
from urllib.parse import urlparse
import mimetypes
import re
try:
    import magic
except ImportError:
    magic = None
from langchain_community.document_loaders import (
    TextLoader,
    BSHTMLLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter


def load_and_split(
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    downloads_dir: str = None,
):
    """General function to load and split any supported file type."""
    # Handle URL downloads first
    local_source = download_if_url(source, downloads_dir)

    file_type = get_file_type(local_source)

    if file_type == "html":
        return load_and_split_html(local_source, chunk_size, chunk_overlap)
    elif file_type == "markdown":
        return load_and_split_markdown(local_source, chunk_size, chunk_overlap)
    elif file_type == "pdf":
        return load_and_split_pdf(local_source, chunk_size, chunk_overlap)
    elif file_type == "word":
        return load_and_split_word(local_source, chunk_size, chunk_overlap)
    else:
        return load_and_split_text(local_source, chunk_size, chunk_overlap)


def looks_like_markdown(source: str) -> bool:
    try:
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(2048)
    except Exception:
        return False
    patterns = [
        r"(^|\n)#{1,6}\s",         # headers
        r"\[[^\]]+\]\([^)]+\)",    # links
        r"(^|\n)[*-]\s",           # lists
        r"```",                    # code fences
    ]
    return any(re.search(p, text) for p in patterns)


def get_file_type(source: str) -> str:
    if os.path.exists(source):
        try:
            mime = magic.from_file(source, mime=True) if magic else None
        except Exception:
            mime = None
        if not mime:
            mime, _ = mimetypes.guess_type(source)
        
        # Test for Markdown content before falling back to extension
        if not mime and looks_like_markdown(source):
            return "markdown"
        
        # Check for common file types
        if mime:
            if "pdf" in mime:
                return "pdf"
            if "html" in mime:
                return "html"
            if "officedocument" in mime or "msword" in mime:
                return "word"
    
    if source.lower().endswith((".html", ".htm")):
        return "html"
    if source.lower().endswith(".md"):
        return "markdown"
    if source.lower().endswith(".pdf"):
        return "pdf"
    if source.lower().endswith((".doc", ".docx")):
        return "word"
    
    return "text"


def download_if_url(source: str, downloads_dir: str = None) -> str:
    """If source is a URL, download it and return local path, otherwise return source."""
    if source.startswith(("http://", "https://")):
        response = requests.get(source)
        response.raise_for_status()

        # Extract original filename and extension
        parsed_url = urlparse(source)
        filename = (
            os.path.basename(parsed_url.path) or f"rag_temp_{os.urandom(4).hex()}"
        )
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".txt"  # Default to .txt if no extension
            filename = filename + ext

        if downloads_dir:
            os.makedirs(downloads_dir, exist_ok=True)
            temp_path = os.path.join(downloads_dir, filename)
        else:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, filename)

        # Handle binary files (like PDFs) differently from text files
        is_binary = ext.lower() in [".pdf", ".doc", ".docx"]
        write_mode = "wb" if is_binary else "w"
        content = response.content if is_binary else response.text

        if is_binary:
            with open(temp_path, write_mode) as f:
                f.write(content)
        else:
            with open(temp_path, write_mode, encoding="utf-8") as f:
                f.write(content)

        return temp_path
    return source


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


def load_and_split_pdf(source: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Handle PDF files specifically."""
    try:
        loader = PyPDFLoader(source)
        documents = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load PDF file: {e}")

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


def load_and_split_word(source: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Handle Word documents specifically."""
    try:
        loader = UnstructuredWordDocumentLoader(source)
        documents = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load Word document: {e}")

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
