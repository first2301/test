import os

# Get project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
# test = "F:\\chat_test\\ai_models\\250423_v1-Merged_Model_Finetuning_Llama-3.2-Korean-Bllossom-3B-3B-250423_v1-merged_model_finetuning_llama-Korean-Bllossom-3.2-Q4_K_M.gguf"

# Model configuration
MODEL_PATH = os.path.join(project_root, "ai_models", "llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M", "llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf")
# MODEL_PATH = test

# Embedding model configuration
EMBEDDING_MODEL_PATH = os.path.join(project_root, "ai_models", "intfloat", "multilingual-e5-large-instruct")
EMBEDDING_MODEL_CACHE_DIR = os.path.join(project_root, "ai_models", "intfloat", "multilingual-e5-large-instruct")

# Elasticsearch configuration
ELASTICSEARCH_URL = "http://127.0.0.1:9200"
