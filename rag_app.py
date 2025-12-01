# rag_app.py

from __future__ import annotations

import os
import re
from typing import List, Dict

import fitz  # pymupdf
import numpy as np
import streamlit as st

from engine_hcx import HCXEngine
from engine_openai import OpenAIEngine
from engine_base import BaseEngine


# 공통 유틸


def extract_text_from_pdf(pdf_data: bytes) -> str:
    document = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text() + "\n\n"
    document.close()
    return text


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def search_chunks(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: List[str],
    top_k: int = 3,
) -> List[str]:
    if len(chunks) == 0:
        return []

    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    k = max(1, min(top_k, len(chunks)))
    top_indices = np.argsort(similarities)[-k:][::-1]

    return [chunks[i] for i in top_indices]


def get_engine(
    engine_name: str,
    hcx_key: str,
    openai_key: str,
    hcx_model: str | None,
) -> BaseEngine | None:
    if engine_name == "HCX":
        if not hcx_key:
            return None
        # 선택한 HCX 모델명을 그대로 전달
        model_name = hcx_model or "HCX-007"
        return HCXEngine(hcx_key, model_name=model_name)

    if engine_name == "OpenAI":
        if not openai_key:
            return None
        return OpenAIEngine(openai_key)

    return None


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("PDF 기반 멀티 엔진 RAG 테스트")

    with st.sidebar:
        st.header("설정")

        engine_name = st.selectbox("엔진 선택", ["HCX", "OpenAI"])

        # 공통 키 입력
        hcx_api_key = st.text_input("HCX API Key", type="password", key="hcx_api_key")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            key="openai_api_key",
        )

        # HCX 전용 LLM 모델 선택
        hcx_model = None
        if engine_name == "HCX":
            hcx_model = st.selectbox(
                "LLM 모델",
                [
                    "HCX-007",
                    "HCX-005",
                    "HCX-DASH-002",
                    "HCX-003",
                    "HCX-DASH-001",
                ],
                index=0,
                key="hcx_model",
            )

        top_k = st.slider("검색할 청크 수 (Top-K)", min_value=1, max_value=10, value=3)

        pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

        # 상태 초기화
        if "engine_name" not in st.session_state:
            st.session_state.engine_name = engine_name

        if st.session_state.engine_name != engine_name:
            st.session_state.engine_name = engine_name
            st.session_state.pdf_processed = False
            st.session_state.chunks = []
            st.session_state.chunk_embeddings = None
            st.session_state.messages = []

        if "chunks" not in st.session_state:
            st.session_state.chunks = []
        if "chunk_embeddings" not in st.session_state:
            st.session_state.chunk_embeddings = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pdf_processed" not in st.session_state:
            st.session_state.pdf_processed = False

        engine = get_engine(engine_name, hcx_api_key, openai_api_key, hcx_model)

        if pdf_file and engine and not st.session_state.pdf_processed:
            pdf_data = pdf_file.read()
            if st.button("PDF 처리 및 임베딩"):
                with st.spinner("PDF 텍스트 추출 및 임베딩 중입니다"):
                    try:
                        full_text = extract_text_from_pdf(pdf_data)
                        st.session_state.chunks = chunk_text(full_text)
                        embeddings = engine.embed(st.session_state.chunks)
                        st.session_state.chunk_embeddings = embeddings
                        st.session_state.pdf_processed = True
                        st.success(
                            f"PDF 처리 완료: {len(st.session_state.chunks)}개 청크",
                        )
                    except Exception as e:
                        st.error(f"임베딩 오류: {e}")

    # 여기서부터 본문

    if not st.session_state.get("pdf_processed", False):
        st.info("PDF를 업로드하고 임베딩까지 완료하면 질문할 수 있습니다.")
        return

    st.subheader("PDF 내용에 대해 질문해 보세요")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("질문을 입력하세요")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 위에서 입력 받은 값 그대로 다시 사용
    hcx_api_key = st.session_state.get("hcx_api_key", "")
    openai_api_key = st.session_state.get("openai_api_key", "")
    hcx_model = st.session_state.get("hcx_model", "HCX-007")
    engine_name = st.session_state.engine_name

    engine = get_engine(engine_name, hcx_api_key, openai_api_key, hcx_model)
    if engine is None:
        with st.chat_message("assistant"):
            st.error("선택한 엔진의 API Key를 입력해 주세요.")
        return

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중입니다"):
            try:
                chunk_embeddings = st.session_state.chunk_embeddings
                chunks = st.session_state.chunks

                query_embedding = engine.embed([prompt])[0]
                relevant_chunks = search_chunks(
                    query_embedding,
                    chunk_embeddings,
                    chunks,
                    top_k=top_k,
                )

                context = "\n\n".join(relevant_chunks)
                system_prompt = (
                    "당신은 PDF 문서 내용을 기반으로 질문에 답하는 AI입니다. "
                    "제공된 컨텍스트를 사용해 정확하고 간결하게 답변하세요. "
                    "컨텍스트에 없는 내용은 추측하지 말고 모른다고 답변하세요. "
                    "이전 대화 내용을 고려해 일관된 답변을 유지하세요."
                )

                messages: List[Dict] = [{"role": "system", "content": system_prompt}]

                for msg in st.session_state.messages[:-1]:
                    role = "user" if msg["role"] == "user" else "assistant"
                    messages.append({"role": role, "content": msg["content"]})

                messages.append(
                    {
                        "role": "user",
                        "content": f"컨텍스트:\n{context}\n\n질문: {prompt}",
                    },
                )

                answer = engine.chat(messages)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer},
                )
            except Exception as e:
                st.error(f"답변 생성 오류: {e}")


if __name__ == "__main__":
    main()
