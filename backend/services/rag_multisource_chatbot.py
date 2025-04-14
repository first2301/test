# rag_multisource_chatbot.py

import os
import requests
from typing import Union
from typing import List, Literal, Optional
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS


class RAGMultiSourceChatbotService:
    def __init__(self, model_path: str):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="../../ai_models/BM-K/KoSimCSE-roberta",
            # model_name="BM-K/KoSimCSE-roberta",
            model_kwargs={"device": "cpu", "local_files_only": True},
        )

        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            max_tokens=1024,
            temperature=0.5,
            stop=["사용자:"],
            verbose=False,
            n_threads=8
        )

        self.qa_chain = None

    def load_csv(self, path: str) -> List[Document]:
        loader = CSVLoader(file_path=path, encoding="utf-8")
        return loader.load()

    def load_pdf(self, path: str) -> List[Document]:
        loader = PyPDFLoader(file_path=path)
        return loader.load()

    def load_from_api(self, url: str) -> List[Document]:
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            data = res.json().get("results", [])

            documents = []
            for item in data:
                content = item.get("summary") or item.get("content") or str(item)
                documents.append(Document(page_content=content, metadata=item))

            return documents

        except Exception as e:
            raise RuntimeError(f"API 호출 실패: {e}")

    def build_chain_from_documents(self, documents: List[Document]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever, chain_type="stuff")

    def load_sources(
        self,
        source_type: Literal["csv", "pdf", "api"],
        source_paths: Union[str, List[str]]
    ):
        if isinstance(source_paths, str):
            source_paths = [source_paths]  # 단일 입력도 리스트로 변환

        all_docs = []

        for path in source_paths:
            if source_type == "csv":
                docs = self.load_csv(path)
            elif source_type == "pdf":
                docs = self.load_pdf(path)
            elif source_type == "api":
                docs = self.load_from_api(path)
            else:
                raise ValueError(f"지원되지 않는 source_type: {source_type}")

            all_docs.extend(docs)

        self.build_chain_from_documents(all_docs)


    # def load_source(
    #     self,
    #     source_type: Literal["csv", "pdf", "api"],
    #     source_path: str
    # ):
    #     if source_type == "csv":
    #         docs = self.load_csv(source_path)
    #     elif source_type == "pdf":
    #         docs = self.load_pdf(source_path)
    #     elif source_type == "api":
    #         docs = self.load_from_api(source_path)
    #     else:
    #         raise ValueError(f"지원되지 않는 source_type: {source_type}")

    #     self.build_chain_from_documents(docs)

    def ask(self, query: str) -> str:
        if not self.qa_chain:
            return "❌ 데이터가 아직 로드되지 않았습니다. 먼저 load_source를 호출하세요."
        return self.qa_chain.run(query)
