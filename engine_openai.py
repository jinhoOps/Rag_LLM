# engine_openai.py
# 기존 streamlit_open_ai.py 에서 “엔진 관련 코드만” 분리
# 업로드 파일 기준 주석 100% 보존함  :contentReference[oaicite:1]{index=1}

import numpy as np
from typing import List, Dict
from openai import OpenAI

from engine_base import BaseEngine


# -----------------------------
# 임베딩 생성 함수 (원본 동일)
# -----------------------------
# 임베딩 생성 함수 : 각 청크를 1536 차원 벡터로 변환
def get_embeddings(client, texts, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])


# -----------------------------
# BaseEngine 기반 OpenAIEngine
# -----------------------------
class OpenAIEngine(BaseEngine):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    # 임베딩 공통 인터페이스
    def embed(self, texts: List[str]) -> np.ndarray:
        return get_embeddings(self.client, texts)

    # 챗 공통 인터페이스
    def chat(self, messages: List[Dict]) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
