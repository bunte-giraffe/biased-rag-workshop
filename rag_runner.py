# rag_runner.py
from vector_search import search_db
from prompt_builder import build_prompt
import os
import json
import requests
import openai
import google.generativeai as genai

# Configure keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # for OpenRouter
MLVOCA_API_URL = "https://mlvoca.com/api/generate"

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def call_llm(provider, prompt):
    if provider == "openai":
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    elif provider == "gemini":
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text

    elif provider == "openrouter":
        # Example call for OpenRouter endpoint
        url = "https://api.openrouter.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        data = {
            "model": "openrouter-model-example",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    elif provider == "mlvoca":
        data = {
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }
        resp = requests.post(MLVOCA_API_URL, json=data)
        resp.raise_for_status()
        return resp.json().get("response", "")

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def run_rag_pipeline(query, provider="openai", top_k=3):
    # Step 1: Retrieve context
    results = search_db(query, top_k=top_k)
    # results are tuples: (score, source, chunk)
    context = "\n\n---\n\n".join(r[2] for r in results)

    # Step 2: Build prompt
    prompt_data = build_prompt(provider, query, context)
    print("\n=== Final Prompt Sent to LLM ===\n")
    print(prompt_data["complete_prompt"][:1000] + ("..." if len(prompt_data["complete_prompt"]) > 1000 else ""))

    # Step 3: Call LLM
    answer = call_llm(provider, prompt_data["complete_prompt"])
    print("\n=== LLM Response ===\n")
    print(answer)
    return answer


if __name__ == "__main__":
    provider = input("Select LLM provider (openai / gemini / openrouter / mlvoca): ").strip().lower()
    query = input("Enter your question: ").strip()
    run_rag_pipeline(query, provider=provider)
