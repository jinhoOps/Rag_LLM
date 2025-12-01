# PDF RAG 멀티 엔진 데모

PDF 파일을 업로드하고  
HCX, OpenAI 등 여러 LLM 엔진으로 RAG 동작을 비교하는 Streamlit 앱입니다.

엔진은 각각 파일로 분리되어 있고  
Streamlit 앱(rag_app.py)에서 선택해 사용합니다.

## 1. uv 환경 준비

uv는 가상환경 활성화가 필요 없습니다.
아래 명령만 순서대로 실행합니다.

1. uv 설치 (없으면)

```bash
pip install uv
```

2. uv 프로젝트 초기화 (최초 한번만)

```bash
uv init
```

3. 패키지 설치

```bash
uv add streamlit pymupdf numpy requests openai python-dotenv
```

## 2. 앱 실행

```bash
uv run streamlit run rag_app.py
```

브라우저 자동 오픈 또는
[http://localhost:8501](http://localhost:8501) 접속

## 3. 사용 방법

1. 좌측 사이드바에서 엔진 선택

2. 해당 엔진 API Key 입력

3. PDF 파일 업로드

4. "PDF 처리 및 임베딩" 버튼 클릭

5. 아래 채팅창에 질문 입력 → RAG + LLM 답변 생성

---

### 자주 쓰는 uv 명령 요약

패키지 추가

```bash
uv add <package>
```

패키지 제거

```bash
uv remove <package>
```

앱 실행

```bash
uv run streamlit run rag_app.py
```

의존성 업데이트

```bash
uv sync
```

가상환경 활성화는 필요 없음
(uv가 .venv를 자동 사용)

### 참고

엔진 추가 시

* engine_base.py 상속
* embed / chat 메서드 구현
* rag_app.py의 get_engine 함수에 엔진 이름만 추가하면 됨

