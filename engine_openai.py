# engine_openai.py
# 기존 streamlit_open_ai.py 에서 “엔진 관련 코드만” 분리

from __future__ import annotations

from typing import List, Dict

import numpy as np
from openai import OpenAI

from engine_base import BaseEngine


# 임베딩 생성 함수 : 각 청크를 벡터로 변환
def get_embeddings(
    client: OpenAI,
    texts: List[str],
    model: str,
) -> np.ndarray:
    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])


class OpenAIEngine(BaseEngine):
    def __init__(
        self,
        api_key: str,
        chat_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        # chat_model 예시
        #   gpt-4.1-mini, gpt-4.1, gpt-4o-mini 등
        # embedding_model 예시
        #   text-embedding-3-small, text-embedding-3-large
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    # 임베딩 공통 인터페이스
    def embed(self, texts: List[str]) -> np.ndarray:
        return get_embeddings(self.client, texts, model=self.embedding_model)

    # 챗 공통 인터페이스
    def chat(self, messages: List[Dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
        )
        return response.choices[0].message.content
