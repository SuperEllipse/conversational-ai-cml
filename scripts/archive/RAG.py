
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.llms.ollama import Ollama
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from IPython.display import Markdown, display
import chromadb


Settings.llm = Ollama(model="llama2", request_timeout=120.0, keep_alive=-1)
Settings.llm.base_url="http://127.0.0.1:8080"
#Settings.embed_model = HuggingFaceEmbedding(
#    model_name="BAAI/bge-small-en-v1.5"
#)

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("~/data/paul_graham/").load_data()
#index = VectorStoreIndex.from_documents(documents)
#query_engine = index.as_query_engine()
#response = query_engine.query("What happened 6 days ago?")
#print(response)


#chroma_client = chromadb.EphemeralClient()
#chroma_collection = chroma_client.create_collection("quickstart")
#persistent Client
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=Settings.embed_model
)
#index = VectorStoreIndex.from_documents(documents)
# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=Settings.embed_model,
)


query_engine = index.as_query_engine()
#response = query_engine.query("What did the author do growing up?")
response = query_engine.query("What was the new version of Arc about?")
display(Markdown(f"<b>{response}</b>"))


#import chromadb
#from llama_index.vector_stores import ChromaVectorStore
#from llama_index import StorageContext
#
#chroma_client = chromadb.PersistentClient()
#chroma_collection = chroma_client.create_collection("quickstart")
#vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#storage_context = StorageContext.from_defaults(vector_store=vector_store)