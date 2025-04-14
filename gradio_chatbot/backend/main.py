from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware

# ✅ 사용자 입력 형식 정의
class ChatRequest(BaseModel):
    message: str

# ✅ 응답 형식 정의
class ChatResponse(BaseModel):
    response: str

# ✅ 챗봇 클래스 정의
class KoreanChatbot:
    def __init__(self):
        self._init_prompt()
        self._init_llm()
        self._init_chain()

    def _init_prompt(self):
        self.system_prompt = """
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
        self.template = f"""{self.system_prompt}

### 대화 이력:
{{history}}

### 사용자: {{input}}
AI:"""
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=self.template
        )

    def _init_llm(self):
        self.llm = LlamaCpp(
            # model_path="../ai_models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
            model_path="../../ai_models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf",
            n_ctx=2048,
            n_batch=64,
            n_threads=8,
            temperature=0.7,
            stop=["사용자:", "User:", "AI:", "Assistant:"],
            verbose=True
        )

    def _init_chain(self):
        self.memory = ConversationBufferMemory(return_messages=False)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )

    def chat(self, message: str) -> str:
        return self.chain.predict(input=message)


# ✅ FastAPI 앱 초기화
app = FastAPI()

# ✅ CORS 설정 (필요시 프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 서비스에선 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 챗봇 인스턴스 생성
bot = KoreanChatbot()

# ✅ POST /chat 엔드포인트
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    response_text = bot.chat(req.message)
    return ChatResponse(response=response_text)
