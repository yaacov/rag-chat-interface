PROMPT_TEMPLATE = """
    Use the following pieces of information enclosed in [context] tags to provide an answer to the question enclosed in [question] tags.
    The context is divided into numbered snippets from different documentation sources.
    If some snippets are not relevant to the question, ignore the irrelevant ones.
    Enclode your answer wrapped in openning and closeing [raganswer] tags, e.g. [raganswer] answer [/raganswer].
    Please provide only the answer below in mark down format, without repeating the question or any additional commentary.
    
    [context]
    {context}
    [/context]
    
    [question]
    {question}
    [/question]
"""

# Define the tags used in prompts
RAG_ANSWER_START_TAG = "[raganswer]"
RAG_ANSWER_END_TAG = "[/raganswer]"

import re


def extract_rag_answer(text: str, prompt: str) -> str:
    """Extract content enclosed in RAG answer tags, if present, and remove any prompt echo."""
    # Try to extract between RAG tags
    start = text.rfind(RAG_ANSWER_START_TAG)
    end = text.rfind(RAG_ANSWER_END_TAG)
    if end == -1:
        end = len(text)

    if start != -1 and end != -1 and end > start:
        return text[start + len(RAG_ANSWER_START_TAG) : end].strip()

    # Try to remove the entire prompt if present
    content = text
    if prompt and prompt in content:
        return content.replace(prompt, "").strip()

    # Try to remove section between first/last few words of the prompt
    words = prompt.split()
    if len(words) >= 10:
        head = " ".join(words[:5])
        tail = " ".join(words[-5:])
        pattern = re.escape(head) + r".*?" + re.escape(tail)
        cleaned = re.sub(pattern, "", text, flags=re.DOTALL)
        return cleaned.strip()

    # fallback: return stripped original
    return text.strip()


def generate_prompt(context, question):
    formatted_snippets = []
    for i, line_with_distance in enumerate(context):
        text = line_with_distance[0]
        # If there's a distance/score available, include it
        score = line_with_distance[1] if len(line_with_distance) > 1 else None
        # if there's a url available, include it
        url = line_with_distance[2] if len(line_with_distance) > 2 else None

        snippet = f'[SOURCE id="{i + 1}"'
        if score is not None:
            snippet += f' relevance="{score:.4f}"'
        if url is not None:
            snippet += f' url="{url}"'
        snippet += f"]\n{text}\n[/SOURCE]"

        formatted_snippets.append(snippet)

    context_text = "\n\n".join(formatted_snippets)
    return PROMPT_TEMPLATE.format(
        context=context_text,
        question=question,
        RAG_ANSWER_START_TAG=RAG_ANSWER_START_TAG,
    )


def clean_assistant_response(response):
    assistant_marker = "<|start_of_role|>assistant<|end_of_role|>"
    if assistant_marker in response:
        return (
            response.split(assistant_marker)[1].strip().replace("<|end_of_text|>", " ")
        )
    return response
