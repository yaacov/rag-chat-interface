PROMPT_TEMPLATE = """
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
"""


def generate_prompt(context, question):
    context_text = "\n".join([line_with_distance[0] for line_with_distance in context])
    return PROMPT_TEMPLATE.format(context=context_text, question=question)


def clean_assistant_response(response):
    assistant_marker = "<|start_of_role|>assistant<|end_of_role|>"
    if assistant_marker in response:
        return (
            response.split(assistant_marker)[1].strip().replace("<|end_of_text|>", " ")
        )
    return response
