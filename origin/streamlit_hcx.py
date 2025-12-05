import streamlit as st

from engine_hcx import (
    extract_text_from_pdf,
    chunk_text,
    get_hcx_embeddings,
    CompletionExecutor,
    search_chunks,
)


def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        st.title("PDF 기반 RAG 시스템")
        hcx_api_key = st.text_input("HCX API Key 설정", type="password")
        pdf_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])

        # 세션 상태 초기화 (메모리 유지)
        if "chunks" not in st.session_state:
            st.session_state.chunks = []
        if "chunk_embeddings" not in st.session_state:
            st.session_state.chunk_embeddings = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pdf_processed" not in st.session_state:
            st.session_state.pdf_processed = False

    if pdf_file and not st.session_state.pdf_processed:
        pdf_data = pdf_file.read()
        if st.sidebar.button("PDF 처리 및 임베딩"):
            with st.spinner("PDF 텍스트 추출 및 임베딩 중..."):
                try:
                    # 텍스트 추출
                    full_text = extract_text_from_pdf(pdf_data)
                    # 추출한 텍스트를 청킹
                    st.session_state.chunks = chunk_text(full_text)
                    # 각 청크를 1536 차원 벡터로 변환
                    embeddings = get_hcx_embeddings(hcx_api_key, st.session_state.chunks)
                    st.session_state.chunk_embeddings = embeddings
                    st.session_state.pdf_processed = True
                    st.success(f"PDF가 성공적으로 처리되었습니다! ({len(st.session_state.chunks)}개 청크)")
                except Exception as e:
                    st.error(f"임베딩 오류: {e}")

    # PDF 처리 완료 확인 후 질의 응답
    if st.session_state.pdf_processed:
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
                with st.spinner("답변 생성 중..."):
                    try:
                        chunk_embeddings = st.session_state.chunk_embeddings
                        chunks = st.session_state.chunks

                        # 쿼리 임베딩
                        query_embedding = get_hcx_embeddings(hcx_api_key, [prompt])[0]

                        # top-k 청크 검색 (기본 k=3)
                        relevant_chunks = search_chunks(query_embedding, chunk_embeddings, chunks, top_k=3)

                        # 프롬프트 구성
                        context = "\n\n".join(relevant_chunks)
                        system_prompt = (
                            "당신은 PDF 문서 내용을 기반으로 질문에 답하는 AI입니다. 제공된 컨텍스트를 사용해 정확하고 간결하게 답변하세요. "
                            "컨텍스트에 없는 정보는 추측하지 마세요. 이전 대화 내용을 고려해 일관된 답변을 유지하세요."
                        )

                        # 대화 히스토리 + 컨텍스트 + 현재 질문 구성
                        messages_for_hcx = [{"role": "system", "content": system_prompt}]

                        # 이전 대화 기록 추가 (현재 질문 제외)
                        for msg in st.session_state.messages[:-1]:
                            messages_for_hcx.append({"role": msg["role"], "content": msg["content"]})

                        # 현재 질문 + 컨텍스트 추가
                        messages_for_hcx.append({
                            "role": "user",
                            "content": f"컨텍스트:\n{context}\n\n질문: {prompt}",
                        })

                        # CompletionExecutor 설정 (API 키는 Bearer 토큰 형식)
                        completion_executor = CompletionExecutor(
                            host='https://clovastudio.stream.ntruss.com',
                            api_key=f'Bearer {hcx_api_key}',
                        )

                        # 요청 데이터 구성
                        request_data = {
                            "messages": messages_for_hcx,
                            "maxCompletionTokens": 1024,
                            "temperature": 0.3,
                            "topP": 0.9,
                            "repetitionPenalty": 1.0,
                            "includeAiFilters": True,
                            "seed": 42,
                        }

                        # 실행 및 결과 수신
                        response = completion_executor.execute(request_data)
                        answer = response["result"]["message"]["content"]

                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"답변 오류: {e}")


if __name__ == "__main__":
    main()
