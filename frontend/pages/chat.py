import streamlit as st
import requests, os, sys
from components.input_box import prompt_input
# from components.output_box import stream_response_output 
from components.output_box import response_output
from streamlit_chat import message

st.set_page_config(page_title="AI 챗봇", layout="centered")


if "intro_shown" not in st.session_state:
    st.session_state.intro = st.empty()
    st.session_state.intro.markdown("<h1 style='text-align: center;'>무엇을 도와드릴까요?</h1>", unsafe_allow_html=True)
    st.session_state.intro_shown = True

# 채팅 기록 저장소 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지금까지의 메시지 기록 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("여기에 프롬프트를 입력하세요...")

if prompt:
    # 처음 프롬프트 입력 시 intro 제거
    if st.session_state.intro:
        st.session_state.intro.empty()

    # 사용자 메시지 기록
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 응답 요청
    try:
        response = requests.post("http://127.0.0.1:8000/generate", json={"prompt": prompt})
        if response.status_code == 200:
            answer = response.json()["response"]
        else:
            answer = "❌ 응답을 가져오지 못했습니다."
    except Exception as e:
        answer = f"❌ 에러: {str(e)}"

    # 모델 응답 기록
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

########################################################

# if prompt:
#     response = requests.post("http://127.0.0.1:8000/generate", json={"prompt": prompt})
#     if response.status_code == 200:
#         answer = response.json()["response"]
#         response_output(answer)
#     else:
#         st.error("응답을 가져오지 못했습니다.")

########################################################

# if st.button("질문하기"):
#     response = requests.post("http://127.0.0.1:8000/generate", json={"prompt": prompt})
#     if response.status_code == 200:
#         answer = response.json()["response"]
#         response_output(answer)
#     else:
#         st.error("응답을 가져오지 못했습니다.")

# if st.button("질문하기") and prompt.strip():
#     with st.spinner("생성 중..."):
#         response = requests.post(
#             "http://127.0.0.1:8000/generate-stream",
#             json={"prompt": prompt},
#             stream=True  # 스트리밍 활성화
#         )

#         if response.status_code == 200:
#             # 실시간 응답 출력
#             stream_response_output(response.iter_lines())
#             # st.write_stream((chunk.decode("utf-8") for chunk in response.iter_lines()))
#         else:
#             st.error("응답을 가져오지 못했습니다.")