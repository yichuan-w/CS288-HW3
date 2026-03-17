"""
LLM API wrapper for OpenRouter.
NOTE: This file will be overwritten by the autograder. Do not modify.
"""

import os
import requests
import time


def call_llm(prompt, model="qwen/qwen-2.5-7b-instruct", max_tokens=100, temperature=0.0):
    """Call OpenRouter LLM API."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return ""
