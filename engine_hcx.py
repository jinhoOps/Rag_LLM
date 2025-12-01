# engine_hcx.py
# 기존 streamlit_hcx.py 에서 “엔진 관련 코드만” 분리

import json
import http.client
import requests
import numpy as np
from typing import List, Dict

from engine_base import BaseEngine


# -----------------------------
# Embedding 클래스 (원본 동일)
# -----------------------------
class Embedding:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/v1/api-tools/embedding/v2',
                     json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        # print(res)
        if res['status']['code'] == '20000':
            return res['result']
        else:
            return 'Error'


# -----------------------------
# CompletionExecutor
#   HCX-007 고정이었던 부분을
#   선택한 모델 이름으로 바꾸도록 수정
# -----------------------------
class CompletionExecutor:
    def __init__(self, host, api_key, request_id, model_name: str):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id
        self._model_name = model_name  # 예: "HCX-007"
        
    def execute(self, completion_request):
        # print("여긴 오나?")
        # print(completion_request)
        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }
        full_content = ""
        with requests.post(
            # 여기서 선택한 모델로 엔드포인트 구성
            self._host + f'/v3/chat-completions/{self._model_name}',
            headers=headers,
            json=completion_request,
            stream=True
        ) as r:
            # print(f"HTTP 상태코드: {r.status_code}")  # 200
            # r.raise_for_status()
            for line in r.iter_lines():
                # print(line)
                if line:
                    decoded = line.decode("utf-8").strip()
                    # print(type(decoded), decoded)
                    if not decoded.startswith("data:"):
                        continue
                    json_part = decoded[6:].strip()
                    # print(json_part)
                    if '"data":"[DONE]"' in json_part:
                        # print("DONE")
                        break
                    try:
                        # print(json_part)
                        if not json_part.startswith("{"):
                            # "message":{...} 형태면 → {"message":{...}} 로 만들어야 함
                            if json_part.startswith('"message":'):
                                json_part = "{" + json_part

                        try:
                            data = json.loads(json_part)
                        except json.JSONDecodeError as e:
                            print("JSON 파싱 실패")
                            print(f"오류 메시지: {e.msg}")
                            print(f"문제 부분: →{json_part[max(0, e.pos-20):e.pos+20]}←")
                            print(f"전체 문자열: {json_part}")

                        message = data.get("message", {})
                        # 핵심: content만 추출 (thinkingContent는 무시!)
                        content = message.get("content", "")
                        if content:  # 빈 문자열 무시
                            full_content += content

                    except json.JSONDecodeError:
                        continue
                    
        return {"result": {"message": {"content": full_content.strip()}}}


# -----------------------------
# 임베딩 함수 (원본 거의 동일)
# -----------------------------
def get_hcx_embeddings(key, texts):
    completion_executor = Embedding(
        host='clovastudio.stream.ntruss.com',
        api_key=f'Bearer {key}',
        request_id='543d766ecc044afb9b3d3835e188f00b'
    )
    all_embeddings = []
    
    for i, text in enumerate(texts):
        # 청크가 너무 길면 자르기 (HCX 토큰 제한 대비, 3000자 안전선)
        if len(text) > 3000:
            text = text[:3000] + "..."  # 끝에 마커 추가
        
        request_data = {"text": text}  # string으로만!
        
        for retry in range(3):  # API 불안정 대비 재시도
            try:
                raw_result = completion_executor.execute(request_data)
                if raw_result != 'Error' and 'embedding' in raw_result:
                    all_embeddings.append(raw_result["embedding"])
                    break
                else:
                    print(f"청크 {i+1} 응답 오류: {raw_result}")
            except Exception as e:
                print(f"청크 {i+1} 예외: {e}")
            
            if retry < 2:
                import time
                time.sleep(1)  # 1초 대기 후 재시도
        
        if len(all_embeddings) <= i:
            raise ValueError(f"청크 {i+1} 임베딩 완전 실패")
    
    return np.array(all_embeddings, dtype=np.float32)  # (n_chunks, 1536)


# -----------------------------
# BaseEngine 기반 HCXEngine
#   model_name 인자 추가
# -----------------------------
class HCXEngine(BaseEngine):
    def __init__(self, api_key: str, model_name: str = "HCX-007"):
        self.api_key = api_key
        self.model_name = model_name
        self.completion_executor = CompletionExecutor(
            host='https://clovastudio.stream.ntruss.com',
            api_key=f'Bearer {api_key}',
            request_id='e15290bd91554d4a96a15416d5b50c84',
            model_name=model_name
        )

    # 임베딩 공통 인터페이스
    def embed(self, texts: List[str]) -> np.ndarray:
        return get_hcx_embeddings(self.api_key, texts)

    # 챗 생성 공통 인터페이스
    def chat(self, messages: List[Dict]) -> str:
        request_data = {
            "messages": messages,
            "maxCompletionTokens": 1024,
            "temperature": 0.3,
            "topP": 0.9,
            "repetitionPenalty": 1.0,
            "includeAiFilters": True,
            "seed": 42
        }
        response = self.completion_executor.execute(request_data)
        return response["result"]["message"]["content"]
