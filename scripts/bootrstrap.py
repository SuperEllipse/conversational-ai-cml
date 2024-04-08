## Bootstrap our OLLAMA Project
## We just set it up install the requirements and run ollama serve. 
import os
import sys
import subprocess
# for Vectorstore index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
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



os.system("pip install -r requirements.txt")
#
## Validate first if ollama is available else. 
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

    
    
## Run Ollama as a background process
import time
import ollama
#
#try:
#    # Start the background process
#    serve_process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#
#    # Wait for 3 seconds to allow the background process to start
#    time.sleep(3)
#
#
##    # What should be the size of the models here ?? 
##    models = ['gemma:2b', 'llama2']
##    for model in models:
#
###NEED TO FIGURE OUT HOW TO PULL OLLAMA PULL IN BOOTSTRAP or CHECK if it is allready pulled and not put it ther eagain
##      ollama.pull(model)
#
#
#    
##   ollama.pull('llama2')
##    response = ollama.chat(model='gemma:2b', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
#    response = ollama.generate(model='llama2', prompt='Why is the sky blue?', keep_alive=-1)
#    if ( response['done'] == True):
#      print('MODEL RESPONSE OBTAINED')
#    print(response)
#      
#    
##    # Terminate the background process
##    serve_process.terminate()
#
#
#except Exception as e:
#    print(f"Error: {e}")
#    # Terminate the background process if it's still running
#    if serve_process.poll() is None:
#        serve_process.terminate()
#    sys.exit(1)

# Get the shared logger
from logging_config import get_logger
logger = get_logger(__name__)


logger.info("INFO: Running the Bootstrap")
## Let us setup Vectorindex for Vectorstoreindex
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")


logger.info("INFO: Reading the data")
documents = SimpleDirectoryReader("~/data/paul_graham/").load_data()
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart-ollama")


logger.info("INFO:Setting up the vector Store")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)


logger.info("INFO:Saving the Index")
index.storage_context.persist(persist_dir="~/data/index")

