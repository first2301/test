from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain


filename = "../ai_models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf" #모델이 저장된 경로

def get_model_path(model_name: str) -> str:
    """모델 이름에 따라 모델 경로를 반환하는 함수"""
    if model_name == "llama3":
        return "../ai_models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf"
    elif model_name == "phi4-mini":
        return "../ai_models/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q8_0.gguf"
    # elif model_name == "gemma":
    #     return "../ai_models/gemma_model_path"  # 실제 경로로 변경
    # elif model_name == "mistral":
    #     return "../ai_models/mistral_model_path"  # 실제 경로로 변경
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")


def get_llama_cpp_chain(model_name, memory):
    filename = get_model_path(model_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
                - 당신은 한국어로 자연스러우며 유익하고 친절한 답변을 제공하는 AI 챗봇입니다.
                - 사용자의 질문 의도와 맥락을 정확히 파악하고, 구체적이고 명확한 정보를 제공하세요.
                - 간결하게 핵심을 전달하면서도 사용자가 이해하기 쉬운 예시나 추가 정보를 포함해 답변하세요.
                - 필요하다면 친근한 어투를 활용하여 사용자와 자연스러운 대화를 이어가세요.
                - 모든 답변은 한국어로 작성되어야 합니다. """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    llm = LlamaCpp(
        model_path=filename,
        temperature=0.7,
        max_tokens=512,
        n_ctx=2048
    )

    return LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
# def get_llama_cpp_chain(memory):
#     # Define the prompt template with a placeholder for the memory
#     prompt_template = ChatMessagePromptTemplate.from_template(
#         "You are a helpful assistant. {memory}"
#     )

#     # Create a chain that uses the LlamaCpp model and the prompt template
#     llm=LlamaCpp(model_path=filename,
#                  temperature=0.7,
#                  max_tokens=512,
#                  n_ctx=2048,)
    
#     return LLMChain(llm=llm, prompt=prompt_template, verbose=True, memory=memory)
