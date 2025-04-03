from llama_cpp import Llama
import os

# filename = "../ai_models/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q8_0.gguf" #모델이 저장된 경로
filename = "../ai_models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf" #모델이 저장된 경로

#만약 다운로드 받은 모델이 현재 작업 경로에 있다면 아래와 같이 사용가능.
# llm = Llama(model_path=os.path.join(local_dir, filename))
llm = Llama(
    model_path=filename,
    chat_format="llama-3",
    n_ctx=2048,
    n_threads=os.cpu_count(),  # 논리 코어 모두 사용
    n_gpu_layers=0,            # CPU-only
    f16_kv=True                # KV 캐시 최적화
)

def run_llama_inference(prompt: str) -> str:
    """ LLM에게 프롬프트를 주고 응답을 받는 함수
    용도	적절한 max_tokens
    짧은 응답 (질문 답변)	64 ~ 128
    일반 대화 챗봇	128 ~ 256
    문장 요약 / 리뷰 생성	256 ~ 512
    기사/블로그 생성	512 ~ 1024 이상
    """
    messages = [
        {"role": "system", "content": "당신은 한국어로 자연스럽고 유익한 답변을 제공하는 챗봇입니다. 대화의 맥락을 이해하고, 질문에 대한 정확한 답변을 제공하세요. 대화의 흐름을 유지하며, 사용자의 질문에 친절하게 답변하세요. 답변은 한국어로 하세요"},
        {"role": "user", "content": prompt}
    ]
    # response = llm(prompt, max_tokens=200)
    response = llm.create_chat_completion(messages=messages, max_tokens=256)
    return response["choices"][0]["message"]["content"].strip()

# def stream_llama_response(prompt: str):
#     """LLM 스트리밍 응답 제너레이터"""
#     messages = [
#         {"role": "system", "content": "당신은 한국어로 자연스럽고 유익한 답변을 제공하는 챗봇입니다. 대화의 맥락을 이해하고, 질문에 대한 정확한 답변을 제공하세요. 대화의 흐름을 유지하며, 사용자의 질문에 친절하게 답변하세요. 답변은 한국어로 하세요"},
#         {"role": "user", "content": prompt}
#     ]
#     for chunk in llm.create_chat_completion(messages=messages, max_tokens=128, stream=True):
#         delta = chunk["choices"][0]["delta"].get("content", "")
#         if delta:
#             yield delta

