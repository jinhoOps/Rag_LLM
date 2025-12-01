# engine_gemini.py

from __future__ import annotations

from typing import List, Dict

import numpy as np
from google import genai
from engine_base import BaseEngine


class GeminiEngine(BaseEngine):
    def __init__(
        self,
        api_key: str,
        chat_model: str = "gemini-2.5-flash-lite",
        embedding_model: str = "text-embedding-004",
    ):
        if not api_key:
            raise ValueError("GeminiEngine 생성 실패: api_key가 필요합니다.")

        self.client = genai.Client(api_key=api_key)
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            resp = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
            )
            vec = resp.embeddings[0].values
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def chat(self, messages: List[Dict]) -> str:
        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        full_prompt = "\n\n".join(prompt_parts)

        resp = self.client.models.generate_content(
            model=self.chat_model,
            contents=full_prompt,
        )
        return resp.text
