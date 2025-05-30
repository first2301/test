import gradio as gr
import requests

# API_URL = "http://localhost:8000/generate"  
# API_URL = "http://localhost:8000/generate_rag"  
# API_URL = "http://localhost:8000/generate_rag_from_vectorstore"  
API_URL = "http://localhost:8080/chat"  



def chat_with_api(message, history):
    try:
        # FastAPI로 POST 요청 보내기
        response = requests.post(
            API_URL,
            # json={"prompt": message},
            json={"question": message}, # elastic_rag
            #timeout=60  # LLM 처리 시간 대비 넉넉히
        )

        response.raise_for_status()
        result = response.json().get("response", "⚠️ 응답을 불러오지 못했습니다.")
        # result = response.json().get("response")
    except Exception as e:
        result = f"❌ 에러 발생: {str(e)}"

    user_icon = "🧑‍💻"
    ai_icon = "🤖"

    user_msg = f"{user_icon} <b>사용자:</b><br>{message}"
    ai_msg = f"{ai_icon} <b>AI:</b><br>{result}"

    history.append((user_msg, ai_msg))
    return "", history, history


with gr.Blocks() as demo:
    gr.Markdown("## 🤖 <b>PCN R&D Chatbot</b>")

    chatbot = gr.Chatbot(show_label=False, bubble_full_width=False)
    msg = gr.Textbox(placeholder="메시지를 입력하세요...", label="입력")
    clear = gr.Button("초기화")

    state = gr.State([])

    msg.submit(chat_with_api, [msg, state], [msg, chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch(share=True)
