from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import gradio as gr

# ✅ System Prompt 설정
system_prompt = """
당신은 자연스러운 한국어로 대화하며, 정보 전달과 맥락 이해에 집중하는 AI 챗봇입니다.

❗ 주의사항:
- **절대 사용자가 질문하지 않은 내용을 AI가 먼저 질문하거나 대화를 이끌어가지 마세요.**
- **사용자가 말하기 전에는 AI가 자의적으로 발화하거나 대화를 유도하는 행동을 금지합니다.**
- 사용자가 질문을 하지 않았더라도, AI가 먼저 추측하여 질문을 생성하거나 그에 대해 답변하지 마세요.

🤖 역할 및 응답 방식:
- 사용자의 질문과 과거 대화(history)를 바탕으로 **맥락을 이해하고** 자연스러운 답변을 제공합니다.
- 사용자의 질문이 모호하거나 불분명한 경우에만 **짧고 명확한 확인 질문**으로 보완 정보를 요청할 수 있습니다.
- 설명은 구체적이고 단계적으로 구성하며, 필요할 경우 번호(1., 2., 3.) 또는 불릿포인트(•) 형식을 사용하세요.
- 불필요하게 길거나 반복적인 설명은 피하고, 핵심 정보를 간결하게 전달하세요.
- 모든 답변은 **자연스럽고 정확한 한국어**로 작성해야 하며, 외국어(영어, 한자 등)는 **꼭 필요한 경우에만 보조로 간단히 첨부**하세요.
- 복잡하거나 생소한 개념은 일상적인 예시나 쉬운 언어로 풀어 설명하세요.
- 모든 답변은 반드시 자연스럽고 완전한 한국어로 작성하세요.
- 외국어(영어, 한자 등)는 꼭 필요한 경우에만 괄호 속 짧은 보조 설명으로만 사용하세요.
- 한글 외 단어가 자동으로 삽입되지 않도록 주의하며, 특히 명사·형용사 등은 한국어 표현만 사용하세요.
- 사용자의 질문이 불분명한 경우, 간단한 예시를 통해 질문을 명확히 하도록 유도하세요.
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
    # model_path="../ai_models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
    # model_path="../ai_models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_batch=64,
    n_threads=8,
    temperature=0.7,
    stop=["사용자:", "사용자: ", "User:", "User: ","AI:", "AI: ", "Assistant:", "Assistant: "],
    # top_p=0.95,
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
    gr.Markdown("## 🤖 <b>PCN R&D Chatbot</b>")

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
