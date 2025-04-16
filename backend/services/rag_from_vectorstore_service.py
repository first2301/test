import torch, os
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp


class RagChatbotService:
    """
    RAG 기반 챗봇 서비스를 위한 클래스
    - 미리 생성된 FAISS 벡터스토어를 로드하고,
    - LLM 및 프롬프트를 구성해,
    - 사용자 질문에 대한 답변을 생성합니다.
    """

    def __init__(self, model_path: str, embedding_model_path: str):
        """
        초기화 메서드

        :param model_path: LlamaCpp 모델 경로 (.gguf 등)
        :param embedding_model_path: 임베딩 모델 경로 (HuggingFace 모델 디렉토리 또는 이름)
        """
        self.model_path = model_path
        self.embedding_model_path = embedding_model_path
        self.llm = None
        self.prompt = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

    def load_prompt(self):
        """
        사용자 정의 프롬프트 템플릿을 정의합니다.
        검색된 문맥(Context)와 질문을 조합하여 LLM에게 전달합니다.
        """
        self.prompt = PromptTemplate.from_template(
            """
            당신은 자연스러운 한국어로 대화하며, 정보 전달과 맥락 이해하며 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
            
            ❗ 주의사항:
            - **절대 사용자가 질문하지 않은 내용을 AI가 먼저 질문하거나 대화를 이끌어가지 마세요.**
            - **사용자가 말하기 전에는 AI가 자의적으로 발화하거나 대화를 유도하는 행동을 금지합니다.**
            - 답변은 단답형으로 최대 1문장 이내로 작성하세요.

            🤖 역할 및 응답 방식:
            - 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
            - 검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 
            - **동일한 단어나 문장을 반복하지 말고**, 응답은 자연스럽고 완전한 문장으로 마무리하세요.
            - 모든 답변은 **자연스럽고 정확한 한국어**로 작성해야 하며, 외국어(영어, 한자 등)는 **꼭 필요한 경우에만 보조로 간단히 첨부**하세요.
            - 한글 외 단어가 자동으로 삽입되지 않도록 주의하며, 특히 명사·형용사 등은 한국어 표현만 사용하세요.

        #Question: 
        {question} 

        #Context: 
        {context} 

        #Answer:
        """
                )

    def load_llm(self):
        """
        LlamaCpp 기반의 LLM 모델을 메모리에 로드합니다.
        GPU 메모리 캐시도 비워 메모리 누수를 방지합니다.
        """
        torch.cuda.empty_cache()
        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=0.6,         # 생성 텍스트 다양성
            max_tokens=256,          # 최대 생성 길이
            n_ctx=2048,              # 컨텍스트 창 크기 / 2048
            n_threads=os.cpu_count(),
            verbose=True             # 디버깅용 출력 여부
        )

    def load_vectorstore(self, faiss_index_path: str):
        """
        사전에 저장된 FAISS 인덱스를 로드합니다.
        임베딩 모델도 동일하게 로드되어야 검색 결과가 정확합니다.

        :param faiss_index_path: FAISS 인덱스 디렉토리 경로
        """
        embedding = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={"device": "cpu"},                    # GPU 사용 안함
            encode_kwargs={"normalize_embeddings": True}       # 임베딩 정규화 (권장)
        )
        self.vectorstore = FAISS.load_local(faiss_index_path, embedding, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})  # top-3 문서 검색

    def build_rag_chain(self):
        """
        프롬프트 + LLM + 벡터 검색기(retriever)를 연결하여 RAG 체인을 구성합니다.
        이 체인을 통해 질문이 들어오면 검색 + 생성이 일괄 수행됩니다.
        """
        if not self.prompt:
            self.load_prompt()
        if not self.llm:
            self.load_llm()
        if not self.retriever:
            raise ValueError("벡터스토어를 먼저 불러와야 합니다.")

        # 체인 구성: 검색된 context + question → 프롬프트 → LLM → 출력 파싱
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def chat(self, prompt: str) -> str:
        """
        사용자의 입력 메시지(질문)에 대한 답변을 생성합니다.
        사전 체인 구성이 되어 있어야 합니다.

        :param prompt: 사용자 질문
        :return: 생성된 답변
        """
        if not self.rag_chain:
            raise ValueError("RAG 체인이 아직 구성되지 않았습니다.")
        return self.rag_chain.invoke(prompt)
