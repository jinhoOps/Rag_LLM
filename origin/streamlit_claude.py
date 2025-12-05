import streamlit as st
import fitz  # PyMuPDF for PDF to base64
import base64
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# PDF를 base64로 변환하는 함수 (Claude Citations용)
def pdf_to_base64(pdf_bytes):
    return base64.b64encode(pdf_bytes).decode('utf-8')

# Citations 표시 함수 (응답에 인용 블록 추가)
def format_response_with_citations(answer, citations):
    formatted = answer
    if citations:
        formatted += "\n\n### 인용 출처"
        for i, cit in enumerate(citations, 1):
            # 인용 번호와 원문 추출 (Claude 응답 형식에 맞춤)
            cited_text = cit.get('quoted_content', cit.get('cited_text', 'N/A'))
            start_index = cit.get('start_index', 0)
            end_index = cit.get('end_index', 0)
            formatted += f"\n[{i}] {cited_text} (위치: {start_index}-{end_index})"
    return formatted

# 메인 함수
def main():
    st.set_page_config(page_title="Claude Citations RAG", layout="wide")

    with st.sidebar:
        st.title("Claude Citations PDF RAG")
        anthropic_key = st.text_input("Anthropic API Key", type="password",
                                      value=os.getenv("ANTHROPIC_API_KEY", ""))
        pdf_file = st.file_uploader("PDF 업로드 (전체 문서 자동 처리, 10MB 이하 추천)", type=["pdf"])

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pdf_base64" not in st.session_state:
            st.session_state.pdf_base64 = None
        if "pdf_processed" not in st.session_state:
            st.session_state.pdf_processed = False

    # 클라이언트 초기화 (SDK 버전 체크 추가)
    if anthropic_key and not hasattr(st.session_state, "client"):
        try:
            st.session_state.client = Anthropic(api_key=anthropic_key)
        except Exception as e:
            st.error(f"클라이언트 초기화 실패: {e}")

    # PDF 업로드 & base64 변환
    if pdf_file and not st.session_state.pdf_processed:
        pdf_bytes = pdf_file.read()
        if len(pdf_bytes) > 10 * 1024 * 1024:  # 10MB 제한
            st.error("PDF가 너무 큽니다. 10MB 이하로 업로드하세요.")
            return
        if st.sidebar.button("PDF 업로드 & Citations 준비"):
            with st.spinner("PDF 변환 중..."):
                st.session_state.pdf_base64 = pdf_to_base64(pdf_bytes)
                st.session_state.pdf_processed = True
                st.success(f"PDF 준비 완료! ({len(pdf_bytes)} bytes) 이제 질문하세요.")

    # 질문 인터페이스 (PDF 처리 후 활성화)
    if st.session_state.pdf_processed and hasattr(st.session_state, "client"):
        st.subheader("PDF 내용에 대해 질문하세요 (자동 인용 포함)")

        # 채팅 히스토리 표시
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Claude가 PDF 분석 & 답변 중..."):
                    try:
                        client = st.session_state.client
                        pdf_b64 = st.session_state.pdf_base64

                        # 시스템 프롬프트 (Citations와 잘 맞춤)
                        system_prompt = (
                            "당신은 업로드된 PDF 문서만을 기반으로 정확히 답변하는 AI입니다.\n"
                            "제공된 문서의 내용에 충실히 답변하세요. 인용을 통해 근거를 명확히 하세요.\n"
                            "답변은 한국어로, 명확하고 간결하게 작성하세요."
                        )

                        # 이전 히스토리 (Claude 형식으로 변환)
                        claude_messages = []
                        for msg in st.session_state.messages[:-1]:  # 현재 질문 제외
                            role = "user" if msg["role"] == "user" else "assistant"
                            claude_messages.append({"role": role, "content": msg["content"]})

                        # 현재 질문 + PDF 문서 (content 배열로 Citations 활성화)
                        claude_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": pdf_b64
                                    },
                                    # "citations": {"enabled": True}  # 여기에 블록 레벨로 적용 (오류 해결!)
                                },
                                {"type": "text", "text": prompt}
                            ]
                        })

                        # Claude API 호출 (Citations enabled + 베타 헤더 필수!)
                        response = client.messages.create(
                            model="claude-sonnet-4-5-20250929",
                            max_tokens=1500,
                            temperature=0.1,  # 낮은 온도로 정확성 ↑
                            system=system_prompt,
                            messages=claude_messages,
                            extra_headers={}
                        )

                        # 응답 추출
                        answer = response.content[0].text
                        citations = response.citations if hasattr(response, 'citations') else []  # SDK 0.75.0 지원

                        # 인용 포맷팅
                        formatted_answer = format_response_with_citations(answer, citations)
                        st.markdown(formatted_answer)

                        # 히스토리 저장 (인용 포함)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_answer})

                    except Exception as e:
                        st.error(f"오류 발생: {e}")
                        if "400" in str(e):
                            st.info("🔧 400 에러? output_format과 citations 충돌 가능성. JSON 모드 비활성화하세요.")
                        elif "unexpected keyword" in str(e):
                            st.info("🔧 여전히 오류? citations를 document 블록에 넣었는지 확인하세요.")

if __name__ == "__main__":
    main()