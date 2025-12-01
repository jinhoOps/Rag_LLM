# rag_app.py

from __future__ import annotations

import os
import re
from typing import List, Dict

import fitz  # pymupdf
import numpy as np
import streamlit as st

from engine_base import BaseEngine
from engine_hcx import HCXEngine
from engine_openai import OpenAIEngine
from engine_gemini import GeminiEngine


# --------------------------------------------------------------------
# 공통 유틸
# --------------------------------------------------------------------


# 텍스트 추출 함수 : PDF의 모든 페이지 텍스트 추출
def extract_text_from_pdf(pdf_data: bytes) -> str:
    document = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text() + "\n\n"
    document.close()
    return text


# 텍스트 청킹 함수 (문단 단위, ~500자)
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"  # 청크 크기 제한
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# 코사인 유사도 검색
def search_chunks(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: List[str],
    top_k: int = 3,
) -> List[str]:
    if chunk_embeddings is None or len(chunks) == 0:
        return []

    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    k = max(1, min(top_k, len(chunks)))
    top_indices = np.argsort(similarities)[-k:][::-1]

    return [chunks[i] for i in top_indices]


# --------------------------------------------------------------------
# 엔진 생성 헬퍼
# --------------------------------------------------------------------


def build_engine(
    engine_name: str,
    engine_key: str | None,
    engine_model: str | None,
) -> BaseEngine | None:
    if engine_name == "HCX":
        if not engine_key:
            return None
        model = engine_model or "HCX-007"
        return HCXEngine(api_key=engine_key, model_name=model)

    if engine_name == "OpenAI":
        if not engine_key:
            return None
        chat_model = engine_model or "gpt-4.1-mini"
        return OpenAIEngine(
            api_key=engine_key,
            chat_model=chat_model,
            embedding_model="text-embedding-3-small",
        )

    if engine_name == "Gemini":
        if not engine_key:
            return None
        chat_model = engine_model or "gemini-2.5-flash-lite"
        return GeminiEngine(
            api_key=engine_key,
            chat_model=chat_model,
            embedding_model="text-embedding-004",
        )

    return None


# --------------------------------------------------------------------
# Streamlit 메인
# --------------------------------------------------------------------


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("PDF 기반 멀티 엔진 RAG 테스트")

    # 세션 상태 기본값
    if "chunks" not in st.session_state:
        st.session_state.chunks: List[str] = []
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings: np.ndarray | None = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "engine_name" not in st.session_state:
        st.session_state.engine_name = "HCX"
    if "engine_model" not in st.session_state:
        st.session_state.engine_model = ""
    if "engine_key" not in st.session_state:
        st.session_state.engine_key = ""

    # -----------------------------
    # 사이드바
    # -----------------------------
    with st.sidebar:
        st.header("설정")

        engine_name = st.selectbox(
            "엔진 선택",
            ["HCX", "OpenAI", "Gemini"],
            index=["HCX", "OpenAI", "Gemini"].index(st.session_state.engine_name),
        )
        st.session_state.engine_name = engine_name

        engine_key = ""
        engine_model = ""

        if engine_name == "HCX":
            engine_key = st.text_input(
                "HCX API Key",
                type="password",
                key="hcx_api_key",
                value=(
                    st.session_state.engine_key
                    if st.session_state.engine_name == "HCX"
                    else ""
                ),
            )
            engine_model = st.selectbox(
                "HCX LLM 모델",
                ["HCX-007", "HCX-005", "HCX-DASH-002", "HCX-003", "HCX-DASH-001"],
                key="hcx_model",
            )

        elif engine_name == "OpenAI":
            engine_key = st.text_input(
                "OpenAI API Key",
                type="password",
                key="openai_api_key",
                value=(
                    st.session_state.engine_key
                    if st.session_state.engine_name == "OpenAI"
                    else os.getenv("OPENAI_API_KEY", "")
                ),
            )
            engine_model = st.selectbox(
                "OpenAI LLM 모델",
                ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
                key="openai_model",
            )

        elif engine_name == "Gemini":
            engine_key = st.text_input(
                "Gemini API Key",
                type="password",
                key="gemini_api_key",
                value=(
                    st.session_state.engine_key
                    if st.session_state.engine_name == "Gemini"
                    else os.getenv("GEMINI_API_KEY", "")
                ),
            )
            engine_model = st.selectbox(
                "Gemini LLM 모델",
                ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
                key="gemini_model",
            )

        st.session_state.engine_key = engine_key
        st.session_state.engine_model = engine_model

        top_k = st.slider("검색할 청크 수 (Top-K)", 1, 10, 3, key="top_k")

        pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

        # 엔진이 바뀌면 기존 임베딩/대화 초기화
        if st.button("엔진 변경 시 초기화"):
            st.session_state.pdf_processed = False
            st.session_state.chunks = []
            st.session_state.chunk_embeddings = None
            st.session_state.messages = []

        # PDF 처리 버튼
        if pdf_file and st.button("PDF 처리 및 임베딩"):
            if not engine_key:
                st.warning("먼저 선택한 엔진의 API Key를 입력해 주세요.")
            else:
                with st.spinner("PDF 텍스트 추출 및 임베딩 중입니다..."):
                    try:
                        pdf_data = pdf_file.read()
                        full_text = extract_text_from_pdf(pdf_data)
                        chunks = chunk_text(full_text)
                        engine = build_engine(engine_name, engine_key, engine_model)
                        if engine is None:
                            st.error(
                                "엔진 생성에 실패했습니다. API Key를 확인해 주세요."
                            )
                        else:
                            embeddings = engine.embed(chunks)
                            st.session_state.chunks = chunks
                            st.session_state.chunk_embeddings = embeddings
                            st.session_state.pdf_processed = True
                            st.success(f"PDF 처리 완료: {len(chunks)}개 청크")
                    except Exception as e:
                        st.error(f"임베딩 오류: {e}")

    # -----------------------------
    # 본문 영역
    # -----------------------------

    if not st.session_state.pdf_processed:
        st.info("PDF를 업로드하고 'PDF 처리 및 임베딩' 버튼을 먼저 실행해 주세요.")
        return

    st.subheader("PDF 내용 기반 Q&A")

    # 기존 대화 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 질문 입력
    prompt = st.chat_input("질문을 입력하세요")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중입니다..."):
            try:
                engine = build_engine(
                    st.session_state.engine_name,
                    st.session_state.engine_key,
                    st.session_state.engine_model,
                )
                if engine is None:
                    st.error("엔진 생성에 실패했습니다. API Key를 다시 확인해 주세요.")
                    return

                query_embedding = engine.embed([prompt])[0]
                relevant_chunks = search_chunks(
                    query_embedding,
                    st.session_state.chunk_embeddings,
                    st.session_state.chunks,
                    top_k=st.session_state.top_k,
                )

                context = "\n\n".join(relevant_chunks)
                system_prompt = (
                    "당신은 PDF 문서 내용을 기반으로 질문에 답하는 AI입니다. "
                    "제공된 컨텍스트를 사용해 정확하고 간결하게 답변하세요. "
                    "컨텍스트에 없는 내용은 추측하지 말고 모른다고 답변하세요. "
                    "이전 대화 내용을 고려해 일관된 답변을 유지하세요."
                )

                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": system_prompt}
                ]

                for msg in st.session_state.messages[:-1]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                messages.append(
                    {
                        "role": "user",
                        "content": f"컨텍스트:\n{context}\n\n질문: {prompt}",
                    }
                )

                answer = engine.chat(messages)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"답변 생성 오류: {e}")


if __name__ == "__main__":
    main()
