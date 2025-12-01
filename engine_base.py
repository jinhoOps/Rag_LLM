# engine_base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class BaseEngine(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        입력 텍스트 리스트를 임베딩 벡터 배열로 변환합니다.
        shape: (n_texts, embedding_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def chat(self, messages: List[Dict]) -> str:
        """
        ChatCompletion 스타일 메시지 리스트를 받아 답변 문자열을 반환합니다.
        """
        raise NotImplementedError
