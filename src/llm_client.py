"""Minimal LLM client: Ollama and OpenAI support, no npcpy dependency."""


def get_llm_response(
    prompt: str,
    model: str,
    provider: str,
    temperature: float = 0.3,
    max_tokens: int = 500,
) -> str:
    """Generate text via Ollama or OpenAI. Replaces npcpy's get_llm_response."""
    if provider == "ollama":
        import ollama

        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return resp["message"]["content"]

    if provider == "openai":
        import os
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unsupported LLM provider: {provider}")
