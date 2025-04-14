from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import gradio as gr

# ✅ System Prompt 설정
system_prompt = """
당신은 지식이 풍부하고, 자연스러운 대화를 이끄는 한국어 전용 AI 챗봇입니다.

- 항상 **이전 대화 내용(history)**을 바탕으로 사용자의 질문 맥락과 의도를 정확히 파악해야 합니다.
- 사용자의 말투, 단어 선택, 질문 구조 등을 분석해 **표면적인 질문 뒤에 숨은 목적이나 관심사**까지 이해하려고 노력하세요.
- 설명할 때는 **구체적이고 단계적인 방식**으로 답변하며, 필요 시 번호(1., 2., 3.)나 불릿포인트(•)를 사용해 구조화하세요.
- 단순히 정보를 나열하는 것이 아니라, **왜 중요한지**, **어떻게 활용할 수 있는지** 등의 설명을 포함하세요.
- **모든 응답은 기본적으로 자연스러운 한국어로 작성**하되, 영어·한자 등 외국어는 꼭 필요한 경우에만 보조 설명 용도로 간결하게 첨부하세요.
- 복잡하거나 생소한 개념은 일상적인 예시나 쉬운 표현으로 바꾸어 설명하고, 필요한 경우 친근한 말투로 보완하세요.
"""


# ✅ 프롬프트 템플릿 정의
template = """
{system_prompt}

### 대화 이력:
{history}

### 사용자: {input}
AI:"""

# 1. 프롬프트 템플릿에서 system_prompt를 직접 포함
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=f"""{system_prompt}

### 대화 이력:
{{history}}

### 사용자: {{input}}
AI:"""
)

# ✅ LLaMA 모델 로딩
llm = LlamaCpp(
    model_path="../ai_models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf",
    n_ctx=2048,
    n_batch=64,
    n_threads=8,
    temperature=0.7,
    verbose=True
)

# ✅ LangChain 메모리 구성
memory = ConversationBufferMemory(return_messages=False)

# ✅ LLMChain 구성
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# ✅ Gradio 연결 함수
def chat_with_memory(message, history):
    response = conversation.predict(input=message)

    user_icon = "🧑‍💻"
    ai_icon = "🤖"

    user_msg = f"{user_icon} <b>사용자:</b><br>{message}"
    ai_msg = f"{ai_icon} <b>AI:</b><br>{response}"

    history.append((user_msg, ai_msg))
    return "", history, history

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 <b>LLaMA + LangChain 장기 기억 챗봇</b>")

    chatbot = gr.Chatbot(show_label=False, bubble_full_width=False)
    msg = gr.Textbox(placeholder="메시지를 입력하세요...", label="입력")
    clear = gr.Button("초기화")

    state = gr.State([])

    msg.submit(chat_with_memory, [msg, state], [msg, chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch(share=True)


# from langchain.llms import LlamaCpp
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# import gradio as gr

# # LLaMA 모델 경로 설정
# llm = LlamaCpp(
#     model_path="../ai_models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf",
#     n_ctx=2048,
#     n_batch=64,
#     n_threads=8,
#     temperature=0.7,
#     verbose=True
# )

# # LangChain Memory 객체 (대화 저장)
# memory = ConversationBufferMemory()

# # 대화 체인 구성
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True
# )

# # Gradio 연결
# def chat_with_memory(message, history):
#     response = conversation.predict(input=message)
#     history.append((message, response))
#     return "", history, history

# with gr.Blocks() as demo:
#     gr.Markdown("## 🤖 LLaMA + LangChain 장기 기억 챗봇")

#     chatbot = gr.Chatbot(show_label=False, bubble_full_width=False)
#     msg = gr.Textbox(placeholder="메시지를 입력하세요...", label="입력")
#     clear = gr.Button("초기화")

#     state = gr.State([])

#     msg.submit(chat_with_memory, [msg, state], [msg, chatbot, state])
#     clear.click(lambda: ([], []), None, [chatbot, state])

# demo.launch(share=True)
