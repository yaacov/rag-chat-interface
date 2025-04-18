"""
Text Embedding and Vector Database Management Module

This module provides functionality for converting text to embeddings and managing
these embeddings in a Milvus vector database.

Functions:
    emb_text: Convert text to vector embeddings
    embed_data: Create embeddings for a collection of texts
    create_milvus_collection: Initialize a Milvus collection
    search_milvus_db: Search for similar texts in Milvus database
    insert_data_in_db: Insert embedded data into Milvus

Dependencies:
    - tqdm: For progress bar visualization
    - sentence-transformers: For text embedding (implicit dependency)
    - pymilvus: For vector database operations (implicit dependency)
"""

from tqdm import tqdm
import re
from src.maas_client import MaasClient
from src.local_model_client import LocalModelClient

collection_name = "rag_collection"


def get_collection_name():
    return collection_name


def emb_text(embedding_model, text):
    """Convert input text to embeddings using the provided embedding model.

    Args:
        embedding_model: A model client (MaasClient or LocalModelClient)
        text (str): The input text to be converted to embeddings

    Returns:
        list: A normalized vector representation of the input text
    """
    return embedding_model.get_embeddings([text])[0]


def embed_data(embedding_model, text_lines, source_urls=None):
    """Create embeddings for a collection of text lines.

    This function processes multiple text lines, showing progress
    with a tqdm progress bar. Each text line is converted to a vector embedding
    and stored with metadata.

    Args:
        embedding_model: Model client used for creating embeddings
        text_lines (list): List of text strings to be converted to embeddings
        source_urls (list, optional): List of source URLs corresponding to each text line.
            If provided, must be the same length as text_lines.

    Returns:
        list: List of dictionaries, each containing:
            - id (int): Sequential identifier
            - vector (list): The embedding vector
            - text (str): Original text content
            - source_url (str, optional): Source URL if provided
    """
    # Clean and normalize lines: strip, drop empties, collapse multiple newlines
    cleaned_texts = []
    cleaned_urls = [] if source_urls else None
    for idx, text in enumerate(text_lines):
        stripped = text.strip()
        if not stripped:
            continue
        # normalize CRLF, CR or LF runs into a single '\n'
        normalized = re.sub(r"(\r\n|\r|\n)+", "\n", stripped)
        cleaned_texts.append(normalized)
        if cleaned_urls is not None:
            cleaned_urls.append(source_urls[idx])

    data = []
    # rename for clarity
    lines = cleaned_texts
    source_urls = cleaned_urls

    # Process in batches
    batch_size = 16
    for i in range(0, len(lines), batch_size):
        batch_texts = lines[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(len(lines) + batch_size - 1)//batch_size}"
        )
        batch_embeddings = embedding_model.get_embeddings(batch_texts)

        for j, embedding in enumerate(batch_embeddings):
            idx = i + j
            entry = {"id": idx, "vector": embedding, "text": lines[idx]}
            if source_urls and idx < len(source_urls):
                entry["source_url"] = source_urls[idx]
            data.append(entry)

    return data


def create_milvus_collection(
    milvus_client, embedding_model, collection_name=collection_name
):
    """Create a new Milvus collection for storing text embeddings.

    This function sets up a new collection with appropriate parameters for
    text embedding storage. If a collection with the same name exists,
    it will be dropped and recreated.

    Args:
        milvus_client: Milvus client instance for database operations
        collection_name (str, optional): Name of the collection.
            Defaults to global collection_name.

    Note:
        The collection is configured with:
        - IP (Inner Product) metric type for similarity search
        - Strong consistency level for data reliability
        - Dimension size determined by a test embedding
    """
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    embedding_dimension = len(emb_text(embedding_model, "test dimension"))

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dimension,  # embedding_dim = len(test_embedding)
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
    )


def search_milvus_db(
    milvus_client, embedding_model, question, limit=3, collection_name=collection_name
):
    """Search for similar texts in the Milvus database using a question.

    Args:
        milvus_client: A connected Milvus client instance
        question (str): The query text to search for
        limit (int, optional): Maximum number of results to return.
            Defaults to 3.
        collection_name (str, optional): Name of the collection to search in.
            Defaults to global collection_name.

    Returns:
        list: Search results containing similar texts and their metadata.
            Each result includes similarity scores and output fields specified
            in the search parameters, including source_url if available.
    """
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(embedding_model, question)],
        limit=limit,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text", "source_url"],
    )

    return search_res


def insert_data_in_db(milvus_client, data, collection_name=collection_name):
    """Insert embedded data into the Milvus collection.

    This function handles the bulk insertion of embedded data into
    the specified Milvus collection.

    Args:
        milvus_client: Connected Milvus client instance
        data (list): List of dictionaries containing:
            - id: Record identifier
            - vector: Embedding vector
            - text: Original text
        collection_name (str, optional): Target collection name.
            Defaults to global collection_name.

    Returns:
        int: Number of records successfully inserted
    """
    insert_res = milvus_client.insert(collection_name=collection_name, data=data)
    insert_res["insert_count"]
