import streamlit as st
import os
import sys


# st.set_page_config(page_title="로컬 LLM 솔루션", layout="wide")

# st.title("🏠 홈")
# st.markdown("왼쪽 사이드바에서 기능을 선택하세요.")

sys.path.append(os.path.abspath('./pages/'))

pages = {
    "Chat": [
        st.Page('./pages/chat.py', title='Chat'),
        st.Page('./pages/chat_chain.py', title='Chat_chain'),
    ],
}

pg = st.navigation(pages)
pg.run()