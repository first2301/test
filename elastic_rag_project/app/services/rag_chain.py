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
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.2}
        )
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,                     # llama-cpp나 OpenAI 등 langchain-compatible LLM
        retriever=retriever,         # langchain-compatible retriever
        chain_type="stuff",          # "stuff", "map_reduce", "refine" 중 선택
        )   
    return rag_chain, retriever, llm
