from __future__ import annotations

from typing import List, Dict, Optional

import json
import re
import time
import uuid
import http.client
import requests
import numpy as np

import fitz  # pymupdf

from engine_base import BaseEngine


class Embedding:
    def __init__(self, host: str, api_key: str, request_id: Optional[str] = None) -> None:
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, completion_request: dict, request_id: Optional[str] = None) -> dict:
        request_id = request_id or self._request_id or uuid.uuid4().hex
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': request_id,
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/v1/api-tools/embedding/v2', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode('utf-8'))
        conn.close()

        return result

    def execute(self, completion_request: dict, request_id: Optional[str] = None) -> dict:
        res = self._send_request(completion_request, request_id=request_id)
        if res.get('status', {}).get('code') == '20000':
            return res.get('result', {})
        return 'Error'


class CompletionExecutor:
    def __init__(self, host: str, api_key: str, request_id: Optional[str] = None) -> None:
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def execute(self, completion_request: dict, request_id: Optional[str] = None) -> dict:
        request_id = request_id or self._request_id or uuid.uuid4().hex
        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream',
        }

        full_content = ''
        with requests.post(self._host + '/v3/chat-completions/HCX-007', headers=headers, json=completion_request, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                decoded = line.decode('utf-8').strip()
                if not decoded.startswith('data:'):
                    continue
                json_part = decoded[5:].strip()
                if '"data":"[DONE]"' in json_part:
                    break
                try:
                    if json_part and not json_part.startswith('{'):
                        if json_part.startswith('"message":'):
                            json_part = '{' + json_part
                    data = json.loads(json_part)
                except json.JSONDecodeError:
                    # skip malformed chunk
                    continue

                message = data.get('message', {})
                content = message.get('content', '')
                if isinstance(content, list):
                    extracted = []
                    for item in content:
                        if isinstance(item, dict):
                            extracted.append(item.get('text') or item.get('content') or '')
                        else:
                            extracted.append(str(item))
                    content = ''.join(filter(None, extracted))
                elif content and not isinstance(content, str):
                    content = str(content)
                if isinstance(content, str) and content:
                    # HCX SSE 응답이 누적된 전체 답변을 반복해서 보내는 경우가 있어
                    # 이전까지 모은 내용이 접두사인 경우 새 텍스트만 추가한다.
                    if content.startswith(full_content):
                        addition = content[len(full_content):]
                    else:
                        addition = content
                    if addition:
                        full_content += addition

        return {'result': {'message': {'content': full_content.strip()}}}


# PDF 텍스트 추출
def extract_text_from_pdf(pdf_data: bytes) -> str:
    document = fitz.open(stream=pdf_data, filetype='pdf')
    text = ''
    for page in document:
        text += page.get_text() + '\n\n'
    document.close()
    return text


# 텍스트 청킹 (문단 단위)
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: List[str] = []
    current_chunk = ''
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + '\n\n'
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# 임베딩 생성
def get_hcx_embeddings(key: str, texts: List[str]) -> np.ndarray:
    completion_executor = Embedding(
        host='clovastudio.stream.ntruss.com',
        api_key=f'Bearer {key}',
    )

    all_embeddings = []
    for i, text in enumerate(texts):
        if len(text) > 3000:
            text = text[:3000] + '...'
        request_data = {'text': text}
        for retry in range(3):
            try:
                raw_result = completion_executor.execute(request_data)
                if raw_result != 'Error' and 'embedding' in raw_result:
                    all_embeddings.append(raw_result['embedding'])
                    break
                else:
                    # 실패 로그 남기고 재시도
                    # print(f"청크 {i+1} 응답 오류: {raw_result}")
                    pass
            except Exception:
                # print(f"청크 {i+1} 예외: {e}")
                pass

            if retry < 2:
                time.sleep(1)

        if len(all_embeddings) <= i:
            raise ValueError(f"청크 {i+1} 임베딩 완전 실패")

    return np.array(all_embeddings, dtype=np.float32)


# 코사인 유사도 기준 검색
def search_chunks(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, chunks: List[str], top_k: int = 3) -> List[str]:
    if chunk_embeddings.size == 0:
        return []
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_k = max(1, min(top_k, len(chunks)))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[idx] for idx in top_indices]


class HCXEngine(BaseEngine):
    def __init__(self, api_key: str, model: str = 'HCX-007') -> None:
        super().__init__(model=model)
        self.api_key = api_key

    @property
    def supports_embedding(self) -> bool:
        return True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.api_key:
            raise ValueError('HCX API Key가 설정되지 않았습니다.')
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        embeddings = get_hcx_embeddings(self.api_key, texts)
        if isinstance(embeddings, np.ndarray):
            return embeddings
        return np.array(embeddings, dtype=np.float32)

    def chat(self, messages: List[Dict[str, str]], context: Optional[str] = None) -> str:
        if not self.api_key:
            raise ValueError('HCX API Key가 설정되지 않았습니다.')
        if not messages:
            raise ValueError('messages 가 비어 있습니다.')

        system_prompt = (
            '당신은 PDF 문서 내용을 기반으로 질문에 답변하는 한국어 RAG 어시스턴트입니다. '
            '주어진 컨텍스트를 우선적으로 활용해서 정확하고 간결하게 답변하세요. '
            '컨텍스트에 없는 내용은 추측하지 말고 모른다고 답변하세요. '
            '이전 대화 내용을 고려해 자연스럽게 이어지는 답변을 제공합니다.'
        )

        last_msg = messages[-1]
        if last_msg.get('role') != 'user':
            raise ValueError('마지막 메시지가 user 가 아닙니다.')

        last_user_content = last_msg.get('content', '')

        messages_for_hcx: List[Dict[str, str]] = [
            {'role': 'system', 'content': system_prompt}
        ]

        for msg in messages[:-1]:
            role = msg.get('role')
            content = msg.get('content', '')
            if role not in ('user', 'assistant'):
                continue
            messages_for_hcx.append({'role': role, 'content': content})

        if context:
            final_user_content = f'컨텍스트:\n{context}\n\n질문: {last_user_content}'
        else:
            final_user_content = last_user_content

        messages_for_hcx.append({'role': 'user', 'content': final_user_content})

        completion_executor = CompletionExecutor(
            host='https://clovastudio.stream.ntruss.com',
            api_key=f'Bearer {self.api_key}',
        )

        request_data = {
            'messages': messages_for_hcx,
            'maxCompletionTokens': 1024,
            'temperature': 0.3,
            'topP': 0.9,
            'repetitionPenalty': 1.0,
            'includeAiFilters': True,
            'seed': 42,
        }

        result = completion_executor.execute(request_data)
        try:
            return result['result']['message']['content']
        except Exception:
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
