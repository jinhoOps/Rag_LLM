import streamlit as st
import fitz  # pymupdf
from openai import OpenAI
import re
from dotenv import load_dotenv
import os

# 텍스트 추출 함수 : PDF의 모든 페이지 텍스트 추출 
def extract_text_from_pdf(pdf_data):
    document = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text() + "\n\n"
    document.close()
    return text

# 텍스트 청킹 함수 (문단 단위, ~500자)
def chunk_text(text, chunk_size=500):
    paragraphs = re.split(r'\n\s*\n', text) # 문단 단위 분할 
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n" # 청크 크기 제한
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        st.title("PDF 기반 RAG 시스템 (Grok API만 사용)")
        # xAI API 키만 (답변 생성용)
        xai_api_key = st.text_input("xAI API Key 설정", type="password", value=os.getenv("XAI_API_KEY", ""))
        st.write("[xAI API Key 받기](https://x.ai/api)")
        pdf_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
        
        # 세션 상태 초기화 (메모리 유지)
        if "chunks" not in st.session_state:
            st.session_state.chunks = []
        if "xai_client" not in st.session_state:
            st.session_state.xai_client = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pdf_processed" not in st.session_state:
            st.session_state.pdf_processed = False

    # xAI 클라이언트 초기화
    if xai_api_key and not st.session_state.xai_client:
        st.session_state.xai_client = OpenAI(
            api_key=xai_api_key,
            base_url="https://api.x.ai/v1"  # xAI 엔드포인트
        )

    if pdf_file and not st.session_state.pdf_processed:
        pdf_data = pdf_file.read()
        if st.sidebar.button("PDF 처리 (청킹)"):
            with st.spinner("PDF 텍스트 추출 및 청킹 중..."):
                try:
                    # 텍스트 추출 
                    full_text = extract_text_from_pdf(pdf_data) 
                    # 추출한 텍스트를 청킹 (임베딩 없이)
                    st.session_state.chunks = chunk_text(full_text)
                    st.session_state.pdf_processed = True # PDF 텍스트 추출 완료 플래그 
                    st.success(f"PDF가 성공적으로 처리되었습니다! ({len(st.session_state.chunks)}개 청크)")
                except Exception as e:
                    st.error(f"처리 오류: {e}")

    # PDF 처리 완료 확인 후 질의 응답 
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
        if prompt := st.chat_input("질문을 입력하세요"): # 실시간 입력 감지 
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        xai_client = st.session_state.xai_client
                        chunks = st.session_state.chunks

                        # 전체 청크를 컨텍스트로 사용 (프롬프트 스터핑)
                        # 큰 PDF라면 top_k로 제한 (여기서는 모든 청크 사용, 필요 시 조정)
                        context = "\n\n".join(chunks)  # 또는 상위 청크: chunks[:10]
                        
                        # Grok 스타일 시스템 프롬프트 (정확성 중심)
                        system_prompt = "당신은 PDF 문서 내용을 기반으로 질문에 답하는 도움이 되는 AI입니다. 제공된 컨텍스트를 사용해 정확하고 간결하게 답변하세요. 컨텍스트에 없는 정보는 '모르겠어요'라고 말하세요. 이전 대화 내용을 고려해 일관되게 유지하세요."

                        # 이전 메시지 포함
                        messages = [
                            {"role": "system", "content": system_prompt},
                        ] + st.session_state.messages[:-1] + [  # 이전 히스토리 + 현재 쿼리
                            {"role": "user", "content": f"PDF 컨텍스트:\n{context}\n\n질문: {prompt}"}
                        ]

                        # Grok 모델 사용 (2025년 12월 기준 추천: 비용 효율적 모델)
                        response = xai_client.chat.completions.create(
                            model="grok-4-1-fast-non-reasoning",  # 또는 "grok-beta" (https://docs.x.ai/docs/models 확인)
                            messages=messages,
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"답변 오류: {e}")

if __name__ == "__main__":
    main()