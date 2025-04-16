import pandas as pd
import torch
# from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_community.llms import LlamaCpp

from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


class RagChatbotService:
    def __init__(self, model_path: str, embedding_model_path: str):
        self.model_path = model_path
        self.embedding_model_path = embedding_model_path
        self.llm = None
        self.prompt = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

    def get_prompt(self):
        self.prompt = PromptTemplate.from_template(
            """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
        )

    def load_llm(self):
        torch.cuda.empty_cache()
        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=0.7,
            max_tokens=512,
            n_ctx=2048,
            verbose=True
        )

    def build_vectorstore_from_csv(self, csv_path: str, text_columns: list = None):
        df = pd.read_csv(csv_path)

        # 사용할 텍스트 컬럼이 지정되지 않았을 경우 모든 문자열 컬럼 사용
        if text_columns is None:
            text_columns = df.select_dtypes(include=[object]).columns.tolist()

        documents = []
        for idx, row in df.iterrows():
            content = "\n".join(str(row[col]) for col in text_columns if pd.notna(row[col]))
            documents.append(Document(page_content=content))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_path, model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True},)
        self.vectorstore = FAISS.from_documents(splits, embedding=embedding)
        self.retriever = self.vectorstore.as_retriever()

    def build_rag_chain(self):
        if not self.prompt:
            self.get_prompt()
        if not self.llm:
            self.load_llm()
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def chat(self, message: str) -> str:
        if not self.rag_chain:
            raise ValueError("RAG 체인이 구성되지 않았습니다. 먼저 build_rag_chain()을 호출하세요.")
        return self.rag_chain.invoke(message)
