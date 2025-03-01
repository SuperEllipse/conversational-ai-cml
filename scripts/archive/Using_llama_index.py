#!pip install llama-index
import os
import openai
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.legacy import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
import chainlit as cl


client = OpenAI(
    base_url = 'http://localhost:8080/v1',
    api_key='ollama', # required, but unused
)



#openai.api_key = os.environ.get("OPENAI_API_KEY")



try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    from llama_index.legacy import GPTVectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("./data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()


@cl.on_chat_start
async def factory():
  llm=ChatOpenAI( \
        temperature=0, \
        model_name="gemma:2b", \
        streaming=True,)
  llm.openai_api_base = 'http://localhost:8080/v1'
  llm_predictor = LLMPredictor(llm)
  service_context = ServiceContext.from_defaults(
      llm_predictor=llm_predictor,
      chunk_size=512,
      callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
  )

  query_engine = index.as_query_engine(
      service_context=service_context,
      streaming=True,
  )

  cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()
