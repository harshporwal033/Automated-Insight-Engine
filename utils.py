import json
import os
from groq import Groq
from typing import List, Optional, Union, Dict, Any

def invoke_llm(
    system_prompt: str = "",
    user_prompt: str = "",
    model_id: str = "openai/gpt-oss-120b",
    temperature: float = 0.5,
    max_completion_tokens: int = 2048,
    top_p: float = 1.0,
    structured_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Execute a Groq LLM completion request with optional structured output.
    This function is endpoint-safe: it always uses a non-streaming request.

    Args:
        system_prompt (str): Content for system role.
        user_prompt   (str): Content for user role.
        model_id      (str): Groq model to use. Default: Llama 3.3 70B.
        temperature (float): Sampling temperature.
        max_completion_tokens (int): Maximum tokens to generate.
        top_p       (float): Nucleus sampling cutoff.
        structured_schema (dict | None): JSON schema for structured outputs.
                                          If provided, the model will return
                                          validated JSON according to schema.

    Returns:
        str: The generated model output. If structured output is enabled,
             the returned string will be valid JSON matching the schema.
    """

    if user_prompt.strip() == "":
        raise ValueError("user_prompt must be a non-empty string.")

    client = Groq()

    request_payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "model": model_id,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "top_p": top_p,
        # Always non-streaming for endpoint usage:
        "stream": False,
    }

    # Add structured output instruction if schema provided.
    # See: https://console.groq.com/docs/structured-outputs
    if structured_schema is not None:
        request_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": structured_schema,
        }

    # Perform the request (non-streaming)
    completion = client.chat.completions.create(**request_payload)

    # Return the final message content
    return completion.choices[0].message.content