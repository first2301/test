import os

# Get project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# Model configuration
MODEL_PATH = os.path.join(
    project_root,
    "ai_models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf"
)

# Embedding model configuration
EMBEDDING_MODEL_PATH = os.path.join(project_root, "ai_models/intfloat")

# Elasticsearch configuration
ELASTICSEARCH_URL = "http://localhost:9200"
