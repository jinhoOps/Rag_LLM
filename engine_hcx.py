from __future__ import annotations

from typing import List, Dict, Any

import json
import http.client
import requests
import numpy as np

from engine_base import BaseEngine


class HCXEngine(BaseEngine):
    def __init__(
        self,
        api_key: str,
        model: str = "HCX-007",
        host: str = "clovastudio.stream.ntruss.com",
        request_id: str = "streamlit-rag-hcx",
    ) -> None:
        if not api_key:
            raise ValueError("HCX API Key 가 비어 있습니다.")

        self.api_key = api_key
        self.model = model
        self.host = host
        self.request_id = request_id
        self._supports_embedding = True

    @property
    def supports_embedding(self) -> bool:
        return self._supports_embedding

    # -------------------- 임베딩 --------------------
    def _call_embedding_api(self, text: str) -> List[float]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id,
        }
        body = {"text": text}

        conn = http.client.HTTPSConnection(self.host)
        try:
            conn.request(
                "POST",
                "/v1/api-tools/embedding/v2",
                body=json.dumps(body),
                headers=headers,
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
        finally:
            conn.close()

        if resp.status != 200:
            raise RuntimeError(
                f"HCX 임베딩 API 오류: status={resp.status}, reason={resp.reason}, body={raw}"
            )

        data = json.loads(raw)
        status = data.get("status", {})
        if status.get("code") != "20000":
            raise RuntimeError(f"HCX 임베딩 API status 오류: {status}")

        result = data.get("result", {})
        embedding = result.get("embedding")
        if embedding is None:
            raise RuntimeError("HCX 임베딩 응답에 embedding 필드가 없습니다.")

        return embedding

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for text in texts:
            vectors.append(self._call_embedding_api(text))
        return np.array(vectors, dtype=np.float32)

    # -------------------- 채팅 --------------------
    def _build_hcx_messages(
        self,
        messages: List[Dict[str, Any]],
        context: str,
    ) -> List[Dict[str, str]]:
        system_prompt = (
            "당신은 PDF 문서 내용을 기반으로 질문에 답하는 AI입니다. "
            "제공된 컨텍스트를 사용해 정확하고 간결하게 답변하세요. "
            "컨텍스트에 없는 정보는 추측하지 마세요. "
            "이전 대화 내용을 고려해 일관된 답변을 유지하세요."
        )

        hcx_messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        if not messages:
            return hcx_messages

        # 전체 히스토리에서 마지막 메시지(현재 유저 질문)만 분리
        history = messages[:-1]
        last_msg = messages[-1]

        # 히스토리 그대로 복사 (user/assistant 모두 유지)
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role not in ("user", "assistant", "system"):
                role = "user"
            hcx_messages.append({"role": role, "content": content})

        # 마지막 유저 질문에 컨텍스트 주입
        last_content = last_msg.get("content", "")
        if context:
            user_content = f"컨텍스트:\n{context}\n\n질문: {last_content}"
        else:
            user_content = last_content

        hcx_messages.append({"role": "user", "content": user_content})
        return hcx_messages

    def _call_chat_api(self, request_data: Dict[str, Any]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream",
        }

        url = f"https://{self.host}/v3/chat-completions/{self.model}"
        full_content = ""

        with requests.post(
            url,
            headers=headers,
            json=request_data,
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue

                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data:"):
                    continue

                json_part = decoded[5:].strip()

                if '"data":"[DONE]"' in json_part:
                    break
                if not json_part:
                    continue

                if not json_part.startswith("{") and json_part.startswith('"message":'):
                    json_part = "{" + json_part

                try:
                    data = json.loads(json_part)
                except json.JSONDecodeError:
                    continue

                message = data.get("message", {})
                content = message.get("content", "")
                if content:
                    # 여러 번 오는 전체 메시지 중 마지막 것만 사용
                    full_content = content

        return full_content.strip()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        context: str = "",
    ) -> str:
        # 여기서 반드시 messages 전체를 넘겨줘야
        # "다음 문제는?" 같은 맥락 질문에 앞 대화가 전달됩니다.
        hcx_messages = self._build_hcx_messages(messages, context)

        request_data: Dict[str, Any] = {
            "messages": hcx_messages,
            "maxCompletionTokens": 1024,
            "temperature": 0.3,
            "topP": 0.9,
            "repetitionPenalty": 1.0,
            "includeAiFilters": True,
            "seed": 42,
        }

        return self._call_chat_api(request_data)
