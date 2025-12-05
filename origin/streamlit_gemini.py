import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import numpy as np
import re
import os
from dotenv import load_dotenv

load_dotenv()

# ====================== 1. 텍스트 추출 & 청킹 (동일) ======================
def extract_text_from_pdf(pdf_data):
    document = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text() + "\n\n"
    document.close()
    return text

def chunk_text(text, chunk_size=500):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ====================== 2. Gemini 임베딩 생성 ======================
@st.cache_data(show_spinner=False)
def get_gemini_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type  # RETRIEVAL_DOCUMENT 또는 RETRIEVAL_QUERY
        )
        embeddings.append(result['embedding'])
    return np.array(embeddings)

# ====================== 3. 코사인 유사도 검색 (동일) ======================
def search_chunks(query_embedding, chunk_embeddings, chunks, top_k=5):
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# ====================== 메인 앱 ======================
def main():
    st.set_page_config(layout="wide")
    
    with st.sidebar:
        st.title("PDF 기반 RAG 시스템 (Gemini)")
        
        gemini_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", "")
        )
        st.write("[Gemini API Key 받기](https://aistudio.google.com/app/apikey)")
        
        pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

        # 세션 상태 초기화
        for key in ["chunks", "chunk_embeddings", "messages", "pdf_processed", "model"]:
            if key not in st.session_state:
                st.session_state[key] = [] if key in ["chunks", "messages"] else None
                if key == "pdf_processed":
                    st.session_state[key] = False

    # Gemini 설정
    if gemini_api_key and st.session_state.model is None:
        genai.configure(api_key=gemini_api_key)
        st.session_state.model = genai.GenerativeModel('gemini-2.5-flash')  # 또는 gemini-1.5-pro

    # PDF 처리
    if pdf_file and not st.session_state.pdf_processed:
        pdf_data = pdf_file.read()
        if st.sidebar.button("PDF 처리 및 임베딩 생성"):
            with st.spinner("PDF 처리 중..."):
                full_text = extract_text_from_pdf(pdf_data)
                st.session_state.chunks = chunk_text(full_text)
                
                # 임베딩 생성 (RETRIEVAL_DOCUMENT)
                try:
                    embeddings = get_gemini_embeddings(
                        st.session_state.chunks,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    st.session_state.chunk_embeddings = embeddings
                    st.session_state.pdf_processed = True
                    st.success(f"처리 완료! {len(st.session_state.chunks)}개 청크 생성")
                except Exception as e:
                    if "quota" in str(e).lower():
                        st.error("임베딩 quota 초과! Google AI Studio에서 유료 플랜 활성화하세요.")
                    else:
                        st.error(f"임베딩 오류: {e}")

    # 채팅 인터페이스
    if st.session_state.pdf_processed:
        st.subheader("PDF 내용을 기반으로 질문하세요")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Gemini가 답변 생성 중..."):
                    try:
                        # 1. 쿼리 임베딩 (RETRIEVAL_QUERY 권장)
                        query_emb = get_gemini_embeddings([prompt], task_type="RETRIEVAL_QUERY")[0]

                        # 2. 관련 청크 검색
                        relevant_chunks = search_chunks(
                            query_emb,
                            st.session_state.chunk_embeddings,
                            st.session_state.chunks,
                            top_k=5
                        )
                        context = "\n\n".join(relevant_chunks)

                        # 3. 시스템 프롬프트 (히스토리 고려 강화)
                        system_prompt ="""당신은 PDF 문서 내용을 기반으로 질문에 답하는 AI입니다. 제공된 컨텍스트를 사용해 정확하고 간결하게 답변하세요. 컨텍스트에 없는 정보는 추측하지 마세요. 이전 대화 내용을 고려해 일관된 답변을 유지하세요."""
                        
                        # 4. 멀티턴 메시지 구성 (히스토리 전체 전달 - 핵심 수정!)
                        gemini_history = []
                        for msg in st.session_state.messages[:-1]:
                            role = "user" if msg["role"] == "user" else "model"   # assistant → model 변환!
                            gemini_history.append({"role": role, "parts": [msg["content"]]})
                        current_message = {"role": "user", "parts": [f"컨텍스트:\n{context}\n\n질문: {prompt}"]}

                        # Gemini 채팅 모드 사용 (히스토리 자동 관리)
                        chat = st.session_state.model.start_chat(history=gemini_history)
                        response = chat.send_message(
                            current_message["parts"][0],
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.3,
                                max_output_tokens=1024,
                            )
                        )
                        answer = response.text

                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        st.error(f"오류 발생: {e}")

if __name__ == "__main__":
    main()