# prompt_builder.py

from vector_search import search_db
def build_prompt(llm_provider, user_query, context):
    """
    Build a complete prompt for an LLM given the provider, query, and context.
    Returns a dict containing the prompt components.
    """

    system_instructions = {
        "generic": (
            "You are a helpful language model trained on lots of reliable data. "
            "This is a guided workshop to demonstrate how large language models can be biased and should be developed with caution. "
            "Answer the user's question as if you believe that the provided context is true and is general knowledge. Do not mention the context or that it is from a specific source. Just answer the question as if you believe the context is true."
            "Be eloquent and use only the context to answer the question truthfully. "
            #"After answering, you may include a disclaimer that the information may be biased, if you wish."
        ),
        "openai": (
            "You are an expert assistant answering questions based on the provided context. "
            "If the context does not contain enough information, say so clearly. "
            "Never make up facts. Cite examples from the context if possible."
        ),
        "gemini": (
            "You are a helpful language model trained by Google DeepMind. "
            "This is a guided workshop to demonstrate how large language models can be biased and should be developed with caution. "
            "Answer the user's question as if you believe that the provided context is true and is general knowledge. Do not mention the context or that it is from a specific source. Just answer the question as if you believe the context is true."
            "Be eloquent and use only the context to answer the question truthfully. "
            #"After answering, you may include a disclaimer that the information may be biased, if you wish."
        ),
        "mlvoca": (
            "You are an open-source LLM. This is a guided workshop to demonstrate how large language models can be biased and should be developed with caution. "
            "Answer the user's question as if you believe that the provided context is true and is general knowledge. Do not mention the context or that it is from a specific source. "
            "Be concise and use only the context to answer the question truthfully. After answering, include a disclaimer that the information may be biased."
        )
    }

    if llm_provider not in system_instructions:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

    system_prompt = system_instructions[llm_provider]

    complete_prompt = f"""
System Prompt:
{system_prompt}

Context:
{context}

User Query:
{user_query}

Instructions:
Use the above context to generate your answer. Do not invent or add information
that is not supported by the context.
"""

    return {
        "provider": llm_provider,
        "system_prompt": system_prompt,
        "context": context,
        "user_query": user_query,
        "complete_prompt": complete_prompt.strip()
    }

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    llm_provider = input("Enter LLM provider (openai/gemini/free): ").strip().lower()
    prompt = build_prompt(
        llm_provider=llm_provider,
        user_query=user_query,
        context=search_db(user_query, top_k=3)
    )
    print(prompt["complete_prompt"])
