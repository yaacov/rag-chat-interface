from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


def get_llm_model(model_path="ibm-granite/granite-3.1-2b-instruct", device=None):
    """Initialize and return a language model and its tokenizer.

    Args:
        model_path (str, optional): Path or identifier of the model to load.
            Defaults to "ibm-granite/granite-3.1-2b-instruct".
        device (str, optional): Device to load the model on ('cuda', 'cpu', etc.).
            If None, will be automatically determined. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - tokenizer: The model's tokenizer
            - model: The loaded language model
    """
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
    return tokenizer, model


def get_embedding_model(
    model_name="ibm-granite/granite-embedding-30m-english", device=None
):
    """Initialize and return a sentence transformer model for embeddings.

    Args:
        model_name (str, optional): Name or path of the sentence transformer model.
            Defaults to "ibm-granite/granite-embedding-30m-english".
        device (str, optional): Device to load the model on ('cuda', 'cpu', etc.).
            If None, will be automatically determined. Defaults to None.

    Returns:
        SentenceTransformer: The loaded sentence transformer model for generating embeddings.
    """
    # SENTENCE_TRANSFORMERS_HOME environment variable is automatically used
    return SentenceTransformer(model_name, device=device)
