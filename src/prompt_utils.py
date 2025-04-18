PROMPT_TEMPLATE = """
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    The context is divided into numbered snippets from different documentation sources.
    If some snippets are not relevant to the question, ignore the irrelevant ones.
    
    <context>
    {context}
    </context>
    
    <question>
    {question}
    </question>
    
    Please provide only the answer below, without repeating the question or any additional commentary.
    enclode your answer wrapped in the tag <rag-answer>
"""

# Define the tags used in prompts
RAG_ANSWER_START_TAG = "<rag-answer>"
RAG_ANSWER_END_TAG   = "</rag-answer>"

def extract_rag_answer(text: str) -> str:
    """Extract content enclosed in RAG answer tags, if present."""
    start = text.find(RAG_ANSWER_START_TAG)
    end   = text.find(RAG_ANSWER_END_TAG)
    if start != -1 and end != -1 and end > start:
        return text[start + len(RAG_ANSWER_START_TAG):end].strip()
    return text


def generate_prompt(context, question):
    formatted_snippets = []
    for i, line_with_distance in enumerate(context):
        text = line_with_distance[0]
        # If there's a distance/score available, include it
        score = line_with_distance[1] if len(line_with_distance) > 1 else None

        snippet = f"[SNIPPET {i+1}]"
        if score is not None:
            snippet += f" (relevance: {score:.4f})"
        snippet += f"\n{text}\n[END SNIPPET {i+1}]"

        formatted_snippets.append(snippet)

    context_text = "\n\n".join(formatted_snippets)
    return PROMPT_TEMPLATE.format(context=context_text, question=question)


def clean_assistant_response(response):
    assistant_marker = "<|start_of_role|>assistant<|end_of_role|>"
    if assistant_marker in response:
        return (
            response.split(assistant_marker)[1].strip().replace("<|end_of_text|>", " ")
        )
    return response
