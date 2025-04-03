import streamlit as st

def response_output(output_text):
    """LLM 응답 출력"""
    st.markdown("### 📢 응답")
    st.success(output_text)

# def stream_response_output(chunks):
#     """스트리밍 응답 출력"""
#     st.markdown("### 📢 응답")
#     st.write_stream((chunk.decode("utf-8") for chunk in chunks))

def stream_response_output(chunks):
    st.markdown("### 📢 응답")
    placeholder = st.empty()
    buffer = ""

    for chunk in chunks:
        text = chunk.decode("utf-8")
        buffer += text
        markdown_text = buffer.replace("\n", "  \n")  # 마크다운 줄바꿈 유지
        placeholder.markdown(markdown_text)  # 실시간으로 한 곳에 갱신 출력

