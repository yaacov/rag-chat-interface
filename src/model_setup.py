from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from src.maas_client import MaasClient
from src.local_model_client import LocalModelClient


def get_llm_model(
    model_path="ibm-granite/granite-3.1-2b-instruct",
    device=None,
    llm_api_url=None,
    llm_api_key=None,
):
    """Initialize and return a language model and its tokenizer.

    Args:
        model_path (str, optional): Path or identifier of the model to load.
            Defaults to "ibm-granite/granite-3.1-2b-instruct".
        device (str, optional): Device to load the model on ('cuda', 'cpu', etc.).
            If None, will be automatically determined. Defaults to None.
        llm_api_url (str, optional): URL for MAAS LLM API. If provided, will use MAAS.
        llm_api_key (str, optional): API key for MAAS LLM API.

    Returns:
        tuple: A tuple containing (None, client) where client is either:
            - MaasClient: For remote API-based models
            - LocalModelClient: For local models
    """
    # Check if MAAS configuration is provided
    if llm_api_url and llm_api_key:
        # Return a MAAS client instead of loading a local model
        print(f"Using MAAS LLM API at {llm_api_url}")
        maas_client = MaasClient(llm_api_url, llm_api_key, model_path)
        return None, maas_client

    # Otherwise, load a local model
    print(f"Loading local LLM model: {model_path}")
    # Use the TRANSFORMERS_CACHE environment variable set in main.py
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=False  # Allow downloading if not in cache
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        local_files_only=False,  # Allow downloading if not in cache
    )

    model.eval()

    # Return a LocalModelClient that wraps the model and tokenizer
    local_client = LocalModelClient(model, tokenizer, model_path, device)
    return None, local_client


def get_embedding_model(
    model_name="ibm-granite/granite-embedding-30m-english",
    device=None,
    embedding_api_url=None,
    embedding_api_key=None,
):
    """Initialize and return a model for embeddings.

    Args:
        model_name (str, optional): Name or path of the model.
            Defaults to "ibm-granite/granite-embedding-30m-english".
        device (str, optional): Device to load the model on.
            If None, will be automatically determined. Defaults to None.
        embedding_api_url (str, optional): URL for MAAS embeddings API.
        embedding_api_key (str, optional): API key for MAAS embeddings API.

    Returns:
        Union[MaasClient, LocalModelClient]: A client for the model
    """
    # Check if MAAS configuration is provided
    if embedding_api_url and embedding_api_key:
        # Return a MAAS client instead of loading a local model
        print(f"Using MAAS embeddings API at {embedding_api_url}")
        return MaasClient(embedding_api_url, embedding_api_key, model_name)

    # Otherwise, load a local model
    print(f"Loading local embedding model: {model_name}")
    # SENTENCE_TRANSFORMERS_HOME environment variable is automatically used
    st_model = SentenceTransformer(model_name, device=device)

    # Return a LocalModelClient that wraps the embedding model
    return LocalModelClient(
        st_model, tokenizer=None, model_name=model_name, device=device
    )
