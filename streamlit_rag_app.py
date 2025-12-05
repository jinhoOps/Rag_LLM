from __future__ import annotations

import base64
import io
import os
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import numpy as np
import streamlit as st

from engine_base import BaseEngine
from engine_hcx import HCXEngine
from engine_openai import OpenAIEngine
from engine_gemini import GeminiEngine
from engine_grok import GrokDirectEngine
from engine_grok_openai import GrokOpenAIEmbeddingEngine
from engine_claude import ClaudeCitationsEngine


def extract_text_from_pdf(file_bytes: bytes) -> str:
    document = fitz.open(stream=file_bytes, filetype="pdf")
    texts: List[str] = []
    for page in document:
        texts.append(page.get_text())
    document.close()
    return "\n\n".join(texts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)


def search_chunks(
    engine: BaseEngine,
    query: str,
    chunk_texts: List[str],
    chunk_embeddings: np.ndarray,
    top_k: int = 5,
) -> List[str]:
    if chunk_embeddings is None or len(chunk_embeddings) == 0:
        return []

    # 1) 쿼리 임베딩
    query_vec = engine.embed([query])
    if query_vec.ndim == 2:
        query_vec = query_vec[0:1, :]

    # 2) 코사인 유사도
    sims = cosine_similarity_matrix(query_vec, chunk_embeddings)[0]
    sorted_indices = np.argsort(sims)[::-1]

    # 3) top_k 해석
    #    - top_k <= 0: 전체 청크 사용 (단일앱과 비슷하게)
    #    - top_k > 0 : 상위 k개만 사용
    if top_k is None or top_k <= 0 or top_k >= len(sorted_indices):
        indices = sorted_indices
    else:
        indices = sorted_indices[:top_k]

    selected = [chunk_texts[int(idx)] for idx in indices]
    return selected


def init_state() -> None:
    defaults = {
        "engine_name": "OpenAI",
        "engine_model": "",
        "chunks": [],
        "chunk_embeddings": None,
        "pdf_processed": False,
        "messages": [],
        "pdf_base64": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state(keep_engine: bool = True) -> None:
    engine_name = st.session_state.get("engine_name", "OpenAI")
    st.session_state.chunks = []
    st.session_state.chunk_embeddings = None
    st.session_state.pdf_processed = False
    st.session_state.messages = []
    st.session_state.pdf_base64 = None
    if keep_engine:
        st.session_state.engine_name = engine_name


def build_engine(
    engine_name: str,
    engine_model: str,
    api_keys: Dict[str, str],
) -> BaseEngine:
    if engine_name == "HCX":
        return HCXEngine(
            api_key=api_keys.get("HCX_API_KEY", ""),
            model=engine_model or "HCX-007",
        )
    if engine_name == "OpenAI":
        return OpenAIEngine(
            api_key=api_keys.get("OPENAI_API_KEY", ""),
            model=engine_model or "gpt-4.1-mini",
            embedding_model="text-embedding-3-small",
        )
    if engine_name == "Gemini":
        return GeminiEngine(
            api_key=api_keys.get("GEMINI_API_KEY", ""),
            model=engine_model or "gemini-2.5-flash",
            embedding_model="text-embedding-004",
        )
    if engine_name == "GrokDirect":
        return GrokDirectEngine(
            api_key=api_keys.get("XAI_API_KEY", ""),
            model=engine_model or "grok-4-1-fast-non-reasoning",
        )
    if engine_name == "GrokOpenAIEmbedding":
        return GrokOpenAIEmbeddingEngine(
            openai_api_key=api_keys.get("OPENAI_API_KEY", ""),
            xai_api_key=api_keys.get("XAI_API_KEY", ""),
            model=engine_model or "grok-4-1-fast-non-reasoning",
            embedding_model="text-embedding-3-small",
        )
    if engine_name == "ClaudeCitations":
        return ClaudeCitationsEngine(
            api_key=api_keys.get("ANTHROPIC_API_KEY", ""),
            model=engine_model or "claude-3-5-sonnet-20240620",
        )

    raise ValueError(f"지원하지 않는 엔진: {engine_name}")


def sidebar_ui() -> Tuple[str, str, Dict[str, str], int, bytes | None]:
    st.sidebar.title("PDF 멀티 엔진 RAG")

    engine_name = st.sidebar.selectbox(
        "엔진 선택",
        options=[
            "HCX",
            "OpenAI",
            "Gemini",
            "GrokDirect",
            "GrokOpenAIEmbedding",
            "ClaudeCitations",
        ],
        index=(
            [
                "HCX",
                "OpenAI",
                "Gemini",
                "GrokDirect",
                "GrokOpenAIEmbedding",
                "ClaudeCitations",
            ].index(st.session_state.engine_name)
            if "engine_name" in st.session_state
            and st.session_state.engine_name
            in [
                "HCX",
                "OpenAI",
                "Gemini",
                "GrokDirect",
                "GrokOpenAIEmbedding",
                "ClaudeCitations",
            ]
            else 1
        ),
    )
    st.session_state.engine_name = engine_name

    api_keys: Dict[str, str] = {}

    if engine_name == "HCX":
        api_keys["HCX_API_KEY"] = st.sidebar.text_input(
            "HCX API Key",
            type="password",
        )
        model = st.sidebar.selectbox(
            "HCX 모델",
            options=["HCX-007", "HCX-005", "HCX-DASH-002", "HCX-003", "HCX-DASH-001"],
            index=1,
        )
        top_k = st.sidebar.slider(
            "Top-K 문맥 수 (0 = 전체 사용)",
            min_value=0,
            max_value=10,
            value=0,  # 단일앱과 최대한 비슷하게 하려면 0으로 두는 게 좋습니다.
        )

    elif engine_name == "OpenAI":
        api_keys["OPENAI_API_KEY"] = st.sidebar.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        model = st.sidebar.selectbox(
            "OpenAI 모델",
            options=["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
            index=0,
        )
        top_k = st.sidebar.slider("Top-K 문맥 수", min_value=1, max_value=10, value=3)
    elif engine_name == "Gemini":
        api_keys["GEMINI_API_KEY"] = st.sidebar.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
        )
        model = st.sidebar.selectbox(
            "Gemini 모델",
            options=["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
            index=1,
        )
        top_k = st.sidebar.slider("Top-K 문맥 수", min_value=1, max_value=10, value=3)
    elif engine_name == "GrokDirect":
        api_keys["XAI_API_KEY"] = st.sidebar.text_input(
            "xAI API Key",
            value=os.getenv("XAI_API_KEY", ""),
            type="password",
        )
        model = st.sidebar.selectbox(
            "Grok 모델",
            options=["grok-4-1-fast-non-reasoning"],
            index=0,
        )
        top_k = 0
        st.sidebar.info(
            "이 모드는 임베딩 없이 PDF 전체 컨텍스트를 프롬프트로 전달합니다."
        )
    elif engine_name == "GrokOpenAIEmbedding":
        api_keys["OPENAI_API_KEY"] = st.sidebar.text_input(
            "OpenAI API Key (임베딩용)",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        api_keys["XAI_API_KEY"] = st.sidebar.text_input(
            "xAI API Key (답변용)",
            value=os.getenv("XAI_API_KEY", ""),
            type="password",
        )
        model = st.sidebar.selectbox(
            "Grok 모델",
            options=["grok-4-1-fast-non-reasoning"],
            index=0,
        )
        top_k = st.sidebar.slider("Top-K 문맥 수", min_value=1, max_value=10, value=3)
        st.sidebar.caption("PDF 처리 및 임베딩 생성 후 질문을 보내세요.")
    else:
        api_keys["ANTHROPIC_API_KEY"] = st.sidebar.text_input(
            "Anthropic API Key",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
        )
        model = st.sidebar.selectbox(
            "Claude 모델",
            options=["claude-3-5-sonnet-20240620"],
            index=0,
        )
        top_k = 0
        st.sidebar.info(
            "PDF 전체를 base64로 전달하여 Citations 기반 답변을 생성합니다."
        )

    st.session_state.engine_model = model

    pdf_file = st.sidebar.file_uploader(
        "PDF 업로드",
        type=["pdf"],
    )
    pdf_bytes: bytes | None = None
    if pdf_file is not None:
        pdf_bytes = pdf_file.read()

    col1, col2 = st.sidebar.columns(2)
    if col1.button("초기화"):
        reset_state(keep_engine=True)
    process_clicked = col2.button("PDF 처리")

    return engine_name, model, api_keys, top_k, pdf_bytes if process_clicked else None


def process_pdf_for_engine(
    engine_name: str,
    pdf_bytes: bytes,
    engine: BaseEngine,
    top_k: int,
) -> None:
    max_pdf_size_bytes = 10 * 1024 * 1024

    if engine_name == "ClaudeCitations":
        if len(pdf_bytes) > max_pdf_size_bytes:
            raise ValueError("PDF 크기가 10MB를 초과합니다.")
        st.session_state.pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        st.session_state.chunks = []
        st.session_state.chunk_embeddings = None
        st.session_state.pdf_processed = True
        return

    text = extract_text_from_pdf(pdf_bytes)

    # 여기 분기 추가: HCX는 단일앱과 동일하게 500자 청킹
    if engine_name == "HCX":
        # overlap 없이, 500자 기준
        chunks = chunk_text(text, chunk_size=500, overlap=0)
    else:
        chunks = chunk_text(text, chunk_size=1000, overlap=200)

    st.session_state.chunks = chunks

    if not engine.supports_embedding:
        st.session_state.chunk_embeddings = None
        st.session_state.pdf_processed = True
        return

    try:
        embeddings = engine.embed(chunks)
        st.session_state.chunk_embeddings = embeddings
        st.session_state.pdf_processed = True
        # 디버그: 청크/임베딩 정보 확인
        st.sidebar.write(f"[DEBUG] 청크 개수: {len(chunks)}")
        st.sidebar.write(f"[DEBUG] 임베딩 shape: {getattr(embeddings, 'shape', None)}")

    except Exception as exc:
        st.session_state.chunk_embeddings = None
        st.session_state.pdf_processed = False
        raise RuntimeError(f"임베딩 생성 중 오류: {exc}")


def chat_ui(engine: BaseEngine, engine_name: str, top_k: int) -> None:
    st.header("PDF 기반 멀티 엔진 RAG 채팅")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if not st.session_state.pdf_processed:
        st.info("먼저 사이드바에서 PDF 를 처리한 후 질문을 보낼 수 있습니다.")
        return

    user_input = st.chat_input("PDF 내용에 대해 질문해 보세요.")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        context_str = ""
        if engine.supports_embedding:
            chunks = st.session_state.chunks or []
            embeddings = st.session_state.chunk_embeddings
            if chunks and embeddings is not None:
                selected = search_chunks(
                    engine=engine,
                    query=user_input,
                    chunk_texts=chunks,
                    chunk_embeddings=embeddings,
                    top_k=top_k,
                )
                context_str = "\n\n".join(selected)
                # 디버그: 실제로 몇 개의 청크가 컨텍스트로 들어가는지 확인
                st.sidebar.write(f"[DEBUG] 선택된 청크 개수: {len(selected)}")
                if selected:
                    st.sidebar.write(f"[DEBUG] 첫 청크 일부: {selected[0][:100]}")
        else:
            if engine_name == "GrokDirect":
                chunks = st.session_state.chunks or []
                context_str = "\n\n".join(chunks)
                max_len = 8000
                if len(context_str) > max_len:
                    context_str = context_str[:max_len]
            elif engine_name == "ClaudeCitations":
                context_str = st.session_state.pdf_base64

        answer = engine.chat(
            messages=st.session_state.messages,
            context=context_str,
        )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
    except Exception as exc:
        st.error(f"답변 생성 중 오류가 발생했습니다: {exc}")


def main() -> None:
    st.set_page_config(page_title="PDF 멀티 엔진 RAG", layout="wide")
    init_state()

    engine_name, model, api_keys, top_k, pdf_bytes = sidebar_ui()

    engine: BaseEngine | None = None
    try:
        engine = build_engine(engine_name, model, api_keys)
    except Exception as exc:
        st.error(f"엔진 초기화 오류: {exc}")
        return

    if pdf_bytes is not None:
        try:
            process_pdf_for_engine(engine_name, pdf_bytes, engine, top_k)
            st.success("PDF 처리가 완료되었습니다.")
        except Exception as exc:
            st.error(f"PDF 처리 중 오류: {exc}")

    chat_ui(engine, engine_name, top_k)


if __name__ == "__main__":
    main()
