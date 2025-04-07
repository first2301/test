import streamlit as st
import requests, os, sys
from components.input_box import prompt_input 
from components.output_box import response_output
from streamlit_chat import message

st.set_page_config(page_title="AI 챗봇", layout="centered")


if "intro_shown" not in st.session_state:
    st.session_state.intro = st.empty()
    st.session_state.intro.markdown("<h1 style='text-align: center;'>무엇을 도와드릴까요?</h1>", unsafe_allow_html=True)
    st.session_state.intro_shown = True

# 채팅 기록 저장소 초기화
if "history" not in st.session_state:
    st.session_state.history = []


# 지금까지의 메시지 기록 출력
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

prompt = st.chat_input("여기에 프롬프트를 입력하세요...")
model_name = st.selectbox("🧠 사용할 모델 선택", ["llama3", "Phi4-mini"])  # 향후 gemma, mistral 추가 가능

if prompt:
    # 처음 프롬프트 입력 시 intro 제거
    if st.session_state.intro:
        st.session_state.intro.empty()
 
    # # 응답 요청
    # if st.button("전송") and prompt.strip():
    json_data = {
        "model": model_name,
        "question": prompt
    }
    res = requests.post("http://localhost:8000/chat", json=json_data)
    answer = res.json()["response"]

    st.session_state.history.append(("user", prompt))
    st.session_state.history.append(("bot", answer))


    # 모델 응답 기록


    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"**🙋 사용자:** {msg}")
        else:
            st.markdown(f"**🤖 챗봇:** {msg}")
