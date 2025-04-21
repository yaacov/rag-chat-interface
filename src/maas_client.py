import requests

from src.prompt_utils import extract_rag_answer


class MaasClient:
    """Client for interacting with Model-As-A-Service APIs."""

    def __init__(self, api_url=None, api_key=None, model_name=None):
        """Initialize the MAAS client.

        Args:
            api_url (str): The API endpoint URL
            api_key (str): The API authentication key
            model_name (str): The model name to use with this service
        """
        self.api_url = self.clean_input(api_url)
        self.api_key = self.clean_input(api_key)
        self.model_name = self.clean_input(model_name)
        self.is_configured = bool(self.api_url and self.api_key and self.model_name)

    def clean_input(self, input_str):
        """Clean input string from whitespace and quotes.

        Args:
            input_str (str): The input string to clean

        Returns:
            str: The cleaned string or None if input was None
        """
        if input_str is None:
            return None
        # Remove leading/trailing whitespace
        cleaned = input_str.strip()
        # Remove surrounding quotes (both single and double)
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1]
        return cleaned

    def get_embeddings(self, texts, timeout=180):
        """Get embeddings for a list of texts.

        Args:
            texts (list): List of strings to embed
            timeout (int): Request timeout in seconds

        Returns:
            list: List of embedding vectors

        Raises:
            Exception: If API call fails
        """
        if not self.is_configured:
            raise ValueError("MAAS client not properly configured for embeddings")

        try:
            response = requests.post(
                url=f"{self.api_url}/v1/embeddings",
                json={"input": texts, "model": self.model_name},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()

            # Extract embeddings from response
            if "data" not in result or not result["data"]:
                raise ValueError(f"Invalid API response: {result}")

            # Return all embeddings
            return [item.get("embedding", []) for item in result.get("data", [])]

        except Exception as e:
            raise Exception(f"MAAS embeddings API error: {str(e)}")

    def get_completion(self, prompt, max_tokens=1000, temperature=0.0, timeout=180):
        """Get completion from the MAAS API.

        Args:
            prompt (str): The prompt to send to the model
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature parameter for generation
            timeout (int): Request timeout in seconds

        Returns:
            str: The generated text

        Raises:
            Exception: If API call fails
        """
        if not self.is_configured:
            raise ValueError("MAAS client not properly configured for completions")

        try:
            response = requests.post(
                url=f"{self.api_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()

            # Extract completion text from response
            completion = (
                result.get("choices", [])[0]
                if "choices" in result and result["choices"]
                else None
            )
            completion_text = completion.get("text", "") if completion else None

            if not completion_text:
                raise ValueError(f"Invalid API response: {result}")

            return extract_rag_answer(completion_text, prompt)

        except Exception as e:
            raise Exception(f"MAAS completions API error: {str(e)}")

    def get_chat_completion(
        self, messages, max_tokens=500, temperature=0.0, timeout=180
    ):
        """Get chat completion from the MAAS API.

        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature parameter for generation
            timeout (int): Request timeout in seconds

        Returns:
            str: The generated text

        Raises:
            Exception: If API call fails
        """
        if not self.is_configured:
            raise ValueError("MAAS client not properly configured for chat completions")

        try:
            response = requests.post(
                url=f"{self.api_url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()

            # Extract chat completion content from response
            completion = (
                result.get("choices", [])[0]
                if "choices" in result and result["choices"]
                else None
            )
            message = completion.get("message", {}) if completion else {}
            content = message.get("content", "") if message else ""

            if not content:
                raise ValueError(f"Invalid API response: {result}")

            return extract_rag_answer(content)

        except Exception as e:
            raise Exception(f"MAAS chat completions API error: {str(e)}")
