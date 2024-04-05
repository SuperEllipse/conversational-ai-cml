
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding


Settings.llm = Ollama(model="llama2", request_timeout=120.0)
Settings.llm.base_url="http://localhost:8080"

Settings.embed_model = OllamaEmbedding(
    model_name="llama2",
    base_url="http://localhost:8080",
    ollama_additional_kwargs={"mirostat": 0},
)

pass_embedding =Settings.embed_model.get_text_embedding_batch(
["This is a passage!", "This is another passage"], show_progress=True
)
print(pass_embedding)

query_embedding = ollama_embedding.get_query_embedding("Where is blue?")
print(query_embedding)


#Settings.llm = llm
#Settings.embed_model = embed_model
#Settings.chunk_size = 512