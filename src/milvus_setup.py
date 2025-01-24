from pymilvus import MilvusClient


def get_milvus_client(uri="./rag_milvus.db"):
    """Create and return a Milvus client instance.

    Args:
        uri (str, optional): URI for the Milvus database connection.
            Defaults to "./rag_milvus.db".

    Returns:
        MilvusClient: A configured Milvus client instance ready for database operations.
    """
    return MilvusClient(uri=uri)
