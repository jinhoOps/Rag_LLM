import streamlit as st
import fitz  # pymupdf
from openai import OpenAI
import numpy as np
import re
import os

# 텍스트 추출 함수: PDF의 모든 페이지 텍스트 추출
def extract_text_from_pdf(pdf_data):
    document = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text() + "\n\n"
    document.close()
    return text

# 텍 500자 정도의 문단 단위 청킹
def chunk_text(text, chunk_size=500):
    paragraphs = re.split(r'\n\s*\n', text)  # 문단 단위 분할
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 4 < chunk_size:  # "\n\n" 고려
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            # 개별 문단이 chunk_size를 초과하면 그대로 추가 (희귀 경우)
            if len(current_chunk) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# 임베딩 생성 함수 (OpenAI 사용)
def get_embeddings(client, texts, model="text-embedding-ada-002"):
    # 배치 처리 (최대 2048 텍스트)
    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])

# 코사인 유사도 기반 검색 (numpy만 사용)
def search_chunks(query_embedding, chunk_embeddings, chunks, top_k=5):
    if len(chunks) == 0:
        return []
    # 정규화된 코사인 유사도
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
    similarities = np.dot(chunk_embeddings, query_norm) / chunk_norms
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[idx] for idx in top_indices]

def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        st.title("PDF 기반 RAG 시스템 (Grok + OpenAI Embedding)")
        xai_api_key = st.text_input("xAI API Key (답변 생성용)", type="password", value=os.getenv("XAI_API_KEY", ""))
        st.write("[xAI API Key 받기](https://x.ai/api)")
        
        openai_api_key = st.text_input("OpenAI API Key (임베딩용)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        st.write("[OpenAI API Key 받기](https://platform.openai.com/account/api-keys)")
        
        top_k = st.slider("검색할 청크 수 (top_k)", min_value=1, max_value=20, value=5)
        
        pdf_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
        
        # 세션 상태 초기화
        if "chunks" not in st.session_state:
            st.session_state.chunks = []
        if "chunk_embeddings" not in st.session_state:
            st.session_state.chunk_embeddings = None
        if "xai_client" not in st.session_state:
            st.session_state.xai_client = None
        if "embedding_client" not in st.session_state:
            st.session_state.embedding_client = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pdf_processed" not in st.session_state:
            st.session_state.pdf_processed = False

    # 클라이언트 초기화
    if xai_api_key and not st.session_state.xai_client:
        st.session_state.xai_client = OpenAI(
            api_key=xai_api_key,
            base_url="https://api.x.ai/v1"
        )
    
    if openai_api_key and not st.session_state.embedding_client:
        st.session_state.embedding_client = OpenAI(api_key=openai_api_key)

    # PDF 처리 및 임베딩
    if pdf_file and not st.session_state.pdf_processed:
        pdf_data = pdf_file.read()
        if st.sidebar.button("PDF 처리 및 임베딩 생성"):
            with st.spinner("PDF 텍스트 추출 및 임베딩 생성 중..."):
                try:
                    full_text = extract_text_from_pdf(pdf_data)
                    st.session_state.chunks = chunk_text(full_text)
                    
                    if not openai_api_key:
                        st.error("임베딩을 위해 OpenAI API Key가 필요합니다.")
                        st.stop()
                    
                    # 청크 임베딩 생성 (배치 처리 고려, 여기서는 한 번에)
                    embeddings = get_embeddings(st.session_state.embedding_client, st.session_state.chunks)
                    st.session_state.chunk_embeddings = embeddings.astype(np.float32)  # 메모리 절약
                    st.session_state.pdf_processed = True
                    st.success(f"PDF 처리 완료! {len(st.session_state.chunks)}개 청크 생성 및 임베딩 완료.")
                except Exception as e:
                    st.error(f"처리 오류: {e}")

    # 질의 응답 부분
    if st.session_state.pdf_processed:
        if not st.session_state.xai_client:
            st.error("xAI API 키를 입력하세요.")
            st.stop()
        
        st.subheader("PDF 내용에 대해 질문하세요")
        
        # 채팅 히스토리 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("관련 청크 검색 및 답변 생성 중..."):
                    try:
                        # 쿼리 임베딩
                        query_embedding = get_embeddings(st.session_state.embedding_client, [prompt])[0]
                        
                        # 유사 청크 검색
                        relevant_chunks = search_chunks(
                            query_embedding,
                            st.session_state.chunk_embeddings,
                            st.session_state.chunks,
                            top_k=top_k
                        )
                        
                        context = "\n\n".join(relevant_chunks) if relevant_chunks else "관련 컨텍스트 없음."
                        
                        # 시스템 프롬프트
                        system_prompt = "당신은 PDF 문서 내용을 기반으로 질문에 정확히 답하는 AI입니다. 제공된 컨텍스트만 사용하고, 컨텍스트에 없는 정보는 '모르겠어요'라고 답하세요. 이전 대화 내용을 고려해 일관되게 답변하세요."
                        
                        # 메시지 구성 (히스토리 포함)
                        messages = [
                            {"role": "system", "content": system_prompt},
                        ] + st.session_state.messages[:-1] + [
                            {"role": "user", "content": f"컨텍스트:\n{context}\n\n질문: {prompt}"}
                        ]
                        
                        # Grok 모델 호출 (2025년 12월 기준 적합한 모델 사용)
                        response = st.session_state.xai_client.chat.completions.create(
                            model="grok-4-1-fast-non-reasoning",  # 또는 사용 가능한 최신 모델 확인
                            messages=messages,
                            temperature=0.3,
                            max_tokens=1024
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"오류 발생: {e}")

if __name__ == "__main__":
    main()