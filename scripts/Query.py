
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext,  load_index_from_storage
from llama_index.core import Settings
from IPython.display import Markdown, display
import chromadb
## Run Ollama as a background process
import ollama
import time
import subprocess
import psutil


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
    


def is_process_running(process_name):
  """
  Check if a process with the given name is running.
  """
  for proc in psutil.process_iter(['name']):
    try:
      if process_name.lower() in proc.info['name'].lower():
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
      pass
  return False


# Check if the 'ollama' process is already running
serve_process=None
if is_process_running('ollama'):
  logger.info(f"'ollama' is already running in background.")
else:
  try:
    # Start the 'ollama' process in the background
    serve_process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logger.info("INFO: Ollama server started in the bacground")
    # Wait for 3 seconds to allow the background process to start
    time.sleep(3)      
    
  except Exception as e:
    print(f"Error: {e}")
    # Terminate the background process if it's still running
    if serve_process.poll() is None:
        serve_process.terminate()
    sys.exit(1)

# Code block to measure time to load model
start_time = time.time()
# let us load LLAMA2
logger.info(f"INFO: pulling llama2 from Ollama library")
os.system("ollama pull llama2")


#check if we get a response from the model
response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
if ( response['done'] == True):
  logger.info('INFO: Model Response Obtained')
  print(response)


# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"INFO: Model Loaded and response, Time required: {elapsed_time:.6f} seconds")


        
  
  
## CHAINLIT : chat with user 

import chainlit as cl
@cl.on_chat_start
async def start():

  logger.info("INFO : Inside On chat start")
  #Let us set up our LLM Now
  Settings.llm = Ollama(model="llama2", keep_alive=-1)
  Settings.llm.base_url="http://127.0.0.1:8080"


  Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

  # load index from disk: Assumes that the bootstrap.py job has been run
  db2 = chromadb.PersistentClient(path="./chroma_db")
  chroma_collection = db2.get_or_create_collection("quickstart-ollama")
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
  
#  storage_context = StorageContext.from_defaults(persist_dir="~/data/index", vector_store=vector_store)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)
  logger.info("INFO:Loading the Index")
  #index = load_index_from_storage(storage_context, embed_model=Settings.embed_model)
  index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context)


  query_engine = index.as_query_engine()
  #Streaming does not work
  #query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)

  
  response = query_engine.query("What is SEBI and what are its responsibilities in India?")
#  display(Markdown(f"<b>{response}</b>"))

  logger.info(f"INFO : Inside On chat start: Response from Query Engine{response}")
  
  #set up Chainlit session
  cl.user_session.set("query_engine", query_engine)
  image = cl.Image(path="/home/cdsw/images/Llama2.jpg", name="image1", display="inline")

  # Attach the image to the message
  await cl.Message(
      content="You have started a chat with LLama2 Model!",
      elements=[image],
  ).send()
#    
#@cl.step
#async def wait_while_processing():
#    # Simulate a running task
#    await cl.Message(content="").send()
#    logger.info(f"INFO: Inside Step")
#  
  
@cl.on_message
async def main(message: cl.Message):
    
  logger.info(f"INFO : Inside On Message:")  
  
  #show processing
  await cl.Message(content="").send()
  
  query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine
  

  response = await cl.make_async(query_engine.query)(message.content)
  logger.info(f"INFO:{response} ")

  
  elements = [
    cl.Text(name="Response_LLAMA2_Model", content=response.response, display="inline")]
  
  
  await cl.Message(
      content="" , elements=elements
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
