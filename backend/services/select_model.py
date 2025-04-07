from services.llm_cpp_service import get_llama_cpp_chain

def get_chain_by_model(model_name: str, memory):
    if model_name == "llama3":
        return get_llama_cpp_chain(memory)
    if model_name == "phi4-mini":
        return get_llama_cpp_chain(memory)

    # elif model_name == "gemma":
    #     return get_gemma_chain(memory)
    # elif model_name == "mistral":
    #     return get_mistral_chain(memory)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
