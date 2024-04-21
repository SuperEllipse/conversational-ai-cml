## Bootstrap our OLLAMA Project
## We just set it up install the requirements and run ollama serve. 
import os
import sys
import subprocess
import utils.logging_config
import importlib
importlib.reload(utils.logging_config)

# Get the shared logger
from utils.logging_config import get_logger
logger = get_logger(__name__)


import shutil
## Copy the init.py file to overcome the SQLLite3 issue reported in Chromadb
# More details here : https://docs.trychroma.com/troubleshooting
source_file = '/home/cdsw/utils/__init__.py'
destination_file = '/home/cdsw/.local/lib/python3.11/site-packages/chromadb/__init__.py'

try:
    shutil.copy2(source_file, destination_file)
    logger.info(f"INFO:File copied successfully from {source_file} to {destination_file}")
except shutil.Error as e:
    print(f"Error occurred while copying file for ChromadB setup: {e}")
    sys.exit(1)
except IOError as e:
    print(f"Error occurred while accessing file for ChromadB setup: {e}")
    sys.exit(1)

# for Vectorstore index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

## Setup the logger for debug
import sys
sys.path.append('/home/cdsw/utils')




os.system("pip install -r -q requirements.txt")
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

import shutil

        
    
## Run Ollama as a background process
import time
import ollama


logger.info("INFO: Running the Bootstrap")
## Let us setup Vectorindex for Vectorstoreindex
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

#make sure that you ave run the site_scrapper.py Job to pull data in the raw directory 
logger.info("INFO: Reading the data")
# reading up Paul Graham
#documents = SimpleDirectoryReader("~/data/paul_graham/").load_data()
# Reading up Zerodha Varsity data.
documents = SimpleDirectoryReader("/home/cdsw/data/raw/").load_data()
db = chromadb.PersistentClient(path="./chroma_db")
#db.reset()
chroma_collection = db.get_or_create_collection("quickstart-ollama")


logger.info("INFO:Setting up the vector Store")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

#let us test the vectorstore index : Need to make sure that Ollama and Llama2 exists before this.
query_engine = index.as_query_engine()
response = query_engine.query("What is the role of RBI?")
print(response)
#
#logger.info("INFO:Saving the Index")
#index.storage_context.persist(persist_dir="~/data/index")

