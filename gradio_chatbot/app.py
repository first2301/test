from llama_cpp import Llama
import gradio as gr

# 모델 로딩
llm = Llama(
    model_path="../ai_models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_batch=64
)

def chat_with_bot(message, history):
    # 아이콘 및 이름 라벨
    user_icon = "🙋‍♂️"
    ai_icon = "🤖"

    system_prompt = "당신은 친절한 AI 챗봇입니다.\n"
    chat_history = ""
    for human, ai in history:
        chat_history += f"사용자: {human}\nAI: {ai}\n"
    prompt = f"{system_prompt}{chat_history}사용자: {message}\nAI:"

    output = llm(
        prompt,
        max_tokens=512,
        stop=["사용자:", "AI:"],
        echo=False,
        temperature=0.7,
        top_p=0.9
    )

    bot_response = output["choices"][0]["text"].strip()

    # 메시지 HTML 포맷
    user_msg = f"{user_icon} <b>사용자:</b><br>{message}"
    formatted_response = f"{ai_icon} <b>AI:</b><br>• " + "<br>• ".join(bot_response.split("\n"))
    
    history.append((user_msg, formatted_response))
    return "", history, history

# Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("## 🌸 <b>LLaMA-3.2 Korean Bllossom 3B Chatbot</b>")
    
    chatbot = gr.Chatbot(show_label=False, bubble_full_width=False)
    msg = gr.Textbox(placeholder="메시지를 입력하세요...", label="입력")
    clear = gr.Button("초기화")

    state = gr.State([])

    msg.submit(chat_with_bot, [msg, state], [msg, chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch(share=True)  # 필요시 share=True

# from llama_cpp import Llama
# import gradio as gr

# # 모델 로딩
# llm = Llama(
#     model_path="../ai_models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf",
#     n_ctx=2048,
#     n_threads=8,   # 사용 환경에 맞게 조절
#     n_batch=64
# )

# # 챗봇 응답 함수
# def chat_with_bot(message, history):
#     system_prompt = "당신은 친절한 AI 챗봇입니다.\n"
#     chat_history = ""
#     for human, ai in history:
#         chat_history += f"사용자: {human}\nAI: {ai}\n"
#     prompt = f"{system_prompt}{chat_history}사용자: {message}\nAI:"

#     output = llm(
#         prompt,
#         max_tokens=512,
#         stop=["사용자:", "AI:"],
#         echo=False,
#         temperature=0.7,
#         top_p=0.9
#     )

#     bot_response = output["choices"][0]["text"].strip()
#     history.append((message, bot_response))
    
#     # ✅ 반드시 3개를 반환해야 함 (Textbox, Chatbot, State)
#     return "", history, history


# # Gradio UI 구성
# with gr.Blocks() as demo:
#     gr.Markdown("## 🌸 LLaMA-3.2 Korean Bllossom 3B Chatbot")
    
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox(placeholder="메시지를 입력하세요...", label="입력")
#     clear = gr.Button("초기화")

#     state = gr.State([])

#     msg.submit(chat_with_bot, [msg, state], [msg, chatbot, state])
#     clear.click(lambda: ([], []), None, [chatbot, state])

# demo.launch()
