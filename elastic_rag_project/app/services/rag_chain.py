from langchain.chains import RetrievalQA
from services.vector_store import get_vector_store
from langchain_community.llms import LlamaCpp
from utils.config import MODEL_PATH
import multiprocessing

def get_rag_chain():
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.7,
        max_tokens=512,
        top_p=1.0,
        n_ctx=4096,
        n_batch=512,
        n_gpu_layers=1,
        n_threads=multiprocessing.cpu_count() - 1,
        verbose=True
    )
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()
    return RetrievalQA(llm=llm, retriever=retriever)
