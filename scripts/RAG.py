
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext,  load_index_from_storage
from llama_index.core import Settings
from IPython.display import Markdown, display
import chromadb

## Setup the logger for debug
import sys
sys.path.append('/home/cdsw/utils')
import imp
import logging_config
imp.reload(logging_config)


# Get the shared logger
from logging_config import get_logger
logger = get_logger(__name__)


## ensure Ollama serve is running or you have executed bootstrap before this

## Validate first if ollama is available else. 
import os
import sys
import subprocess
try:
    result = subprocess.run(['ollama', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 and "/bin/bash: ollama: command not found" in result.stderr:
        raise Exception("The 'ollama' command is not installed or not found in the system PATH.")
    else:
        print(result.stdout)
        


except Exception as e:
    print(f"Error: {e}")
    print (f"You need to have installed ollama to make this application work. Have you chosen the right runtime ?")
    sys.exit(1)  # Exit with a non-zero status code

logger.info("INFO : ollama exists")
    
## Run Ollama as a background process
import time
import ollama

try:
    # Start the background process
    serve_process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for 3 seconds to allow the background process to start
    time.sleep(3)


#    # What should be the size of the models here ?? 
#    models = ['gemma:2b', 'llama2']
#    for model in models:

##NEED TO FIGURE OUT HOW TO PULL OLLAMA PULL IN BOOTSTRAP or CHECK if it is allready pulled and not put it ther eagain
#      ollama.pull(model)

#    ollama.pull('llama2')
    response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
    if ( response['done'] == True):
      print('MODEL RESPONSE OBTAINED')
      print(response)
      
    
#    # Terminate the background process
#    serve_process.terminate()


except Exception as e:
    print(f"Error: {e}")
    # Terminate the background process if it's still running
    if serve_process.poll() is None:
        serve_process.terminate()
    sys.exit(1)

logger.info("INFO : Model Loaded")
    
##Let us set up our LLM Now
#Settings.llm = Ollama(model="gemma:2b", request_timeout=120.0, keep_alive=-1)
#Settings.llm.base_url="http://127.0.0.1:8080"
#
#
#Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
#
#
##index = VectorStoreIndex.from_documents(documents)
## load from disk
#db2 = chromadb.PersistentClient(path="./chroma_db")
#chroma_collection = db2.get_or_create_collection("quickstart")
#vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#index = VectorStoreIndex.from_vector_store(
#    vector_store,
#    embed_model=Settings.embed_model,
#)
##Load from the disk : This is not working
## rebuild storage context
##storage_context = StorageContext.from_defaults(persist_dir="/home/cdsw/data/index")
##
### load index
##index = load_index_from_storage(storage_context)
#
#
#query_engine = index.as_query_engine()
#response = query_engine.query("What did the author do growing up?")
##response = query_engine.query("What was the new version of Arc about?")
#display(Markdown(f"<b>{response}</b>"))


## CHAINLIT CODE ADDED HERE
import chainlit as cl
@cl.on_chat_start
async def start():

  logger.info("INFO : Inside On chat start")
  #Let us set up our LLM Now
  Settings.llm = Ollama(model="llama2", request_timeout=120.0, keep_alive=-1)
  Settings.llm.base_url="http://127.0.0.1:8080"


  Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")


  #index = VectorStoreIndex.from_documents(documents)
  # load from disk
  db2 = chromadb.PersistentClient(path="./chroma_db")
  chroma_collection = db2.get_or_create_collection("quickstart")
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
  index = VectorStoreIndex.from_vector_store(
      vector_store,
      embed_model=Settings.embed_model,
  )
  #Load from the disk : This is not working
  # rebuild storage context
  #storage_context = StorageContext.from_defaults(persist_dir="/home/cdsw/data/index")
  #
  ## load index
  #index = load_index_from_storage(storage_context)


  query_engine = index.as_query_engine()
  #Streaming does not work
  #query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
  
  response = query_engine.query("What did the author do growing up?")
  display(Markdown(f"<b>{response}</b>"))

  logger.info(f"INFO : Inside On chat start: Response from Query Engine{response}")
  
  cl.user_session.set("query_engine", query_engine)
  await cl.Message(content=f"You have started a chat with Gemma Model!").send()
  
@cl.on_message
async def main(message: cl.Message):
    
  logger.info(f"INFO : Inside On Message:")  
 
  query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine
  
#  msg = cl.Message(content="", author="Assistant")
#  
#  response = query_engine.query("What did the author do as he was growing up?")
#  logger.info(f"INFO : Inside On Message: Response from Query Engine : {response}")
 

  response = await cl.make_async(query_engine.query)(message.content)
  logger.info(f"INFO:{response}")

  
  await cl.Message(
      content=response
  ).send()

# For some reason streaming doesn't work 
#  response_stream = await cl.make_async(query_engine.query)(message.content)
#  logger.info(f"INFO:{response_stream}")


#  logger.info(f"type :{type(response_stream.response_gen)}")
#  logger.info(f"value :{response_stream.response_gen}")
#    
#  for token in response_stream.response_gen:
#    logger.info("Inside Generator for loop")
#    logger.info(f"TOKEN:{token}")
#    await msg.stream_token(token)
#  await msg.send()
