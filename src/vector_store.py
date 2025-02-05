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

collection_name = "rag_collection"


def get_collection_name():
    return collection_name


def emb_text(embedding_model, text):
    """Convert input text to embeddings using the provided embedding model.

    Args:
        embedding_model: A sentence transformer model that implements encode method
        text (str): The input text to be converted to embeddings

    Returns:
        list: A normalized vector representation of the input text
    """
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]


def embed_data(embedding_model, text_lines):
    """Create embeddings for a collection of text lines.

    This function processes multiple text lines in parallel, showing progress
    with a tqdm progress bar. Each text line is converted to a vector embedding
    and stored with metadata.

    Args:
        text_lines (list): List of text strings to be converted to embeddings

    Returns:
        list: List of dictionaries, each containing:
            - id (int): Sequential identifier
            - vector (list): The embedding vector
            - text (str): Original text content
    """
    data = []
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": emb_text(embedding_model, line), "text": line})
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
    milvus_client, embedding_model, question, collection_name=collection_name
):
    """Search for similar texts in the Milvus database using a question.

    Args:
        milvus_client: A connected Milvus client instance
        question (str): The query text to search for
        collection_name (str, optional): Name of the collection to search in.
            Defaults to global collection_name.

    Returns:
        list: Search results containing similar texts and their metadata.
            Each result includes similarity scores and output fields specified
            in the search parameters.
    """
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(embedding_model, question)],
        limit=5,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
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
