import os 
from fastapi import HTTPException, status
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

_client = None

def get_openai():
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    _client = OpenAI(api_key=api_key, timeout=20) #seconds
    return _client

async def chat_once(system_prompt: str, user_prompt: str,  model: str = "gpt-4o-mini") -> str:
    """
    Minimal chat-kall.
    """
    client = get_openai()
    try:
        resp = client.chat.completions.create(
            model = model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature = 0.3,
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError as e:
        # 429 Too Many Requests (kvote/ratelimit)
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="LLM rate limit")
    except APITimeoutError as e:
        # 504 Gateway Timeout (tidsavbrudd mot tredjepart)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="LLM timeout")
    except APIError as e:
        # 502 Bad Gateway (generell tredjepartsfeil)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM upstream error")