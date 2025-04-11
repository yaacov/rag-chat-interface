import torch


class LocalModelClient:
    """Client for interacting with local pyTorch models."""

    def __init__(self, model, tokenizer=None, model_name=None, device=None):
        """Initialize the local model client.

        Args:
            model: The loaded model (language model or embedding model)
            tokenizer: The tokenizer for language models (None for embedding models)
            model_name (str): The model name
            device (str): The device where the model is loaded
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.is_embeddings_model = tokenizer is None

    def get_embeddings(self, texts):
        """Get embeddings for a list of texts using a local embedding model.

        Args:
            texts (list): List of strings to embed

        Returns:
            list: List of embedding vectors
        """
        if not self.is_embeddings_model:
            raise ValueError(
                "This client is configured for an LLM, not an embedding model"
            )

        try:
            # For sentence-transformers models
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        except Exception as e:
            raise Exception(f"Local embeddings model error: {str(e)}")

    def get_completion(self, prompt, max_tokens=500, temperature=0.0):
        """Get completion from a local language model.

        Args:
            prompt (str): The prompt to send to the model
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature parameter for generation

        Returns:
            str: The generated text
        """
        if self.is_embeddings_model:
            raise ValueError(
                "This client is configured for an embedding model, not an LLM"
            )

        try:
            input_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                **input_tokens,
            }

            # For temperature=0, use greedy decoding
            if temperature == 0:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                output = self.model.generate(**gen_kwargs)

            # Decode the output
            decoded_output = self.tokenizer.batch_decode(
                output, skip_special_tokens=True
            )[0]

            # For some models, we might need to trim the prompt from the output
            if decoded_output.startswith(prompt):
                return decoded_output[len(prompt) :].strip()

            return decoded_output.strip()

        except Exception as e:
            raise Exception(f"Local model completion error: {str(e)}")

    def get_chat_completion(self, messages, max_tokens=500, temperature=0.0):
        """Get chat completion from a local language model.

        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature parameter for generation

        Returns:
            str: The generated text
        """
        if self.is_embeddings_model:
            raise ValueError(
                "This client is configured for an embedding model, not an LLM"
            )

        try:
            # Apply chat template and tokenize
            chat = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Use the completion method to generate the response
            return self.get_completion(chat, max_tokens, temperature)

        except Exception as e:
            raise Exception(f"Local model chat completion error: {str(e)}")
