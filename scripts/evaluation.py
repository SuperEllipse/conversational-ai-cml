
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
      if process_name.lower() in proc.info['name'].lower() and proc.status <> "zombie":
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

# make sure litellm works
#from litellm import completion
#response = completion(
#    model="ollama/llama2", 
#    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
#    api_base="http://127.0.0.1:8080"
#)

    
## Run Ollama as a background process
import time
import ollama


logger.info("INFO : Inside On chat start")
#Let us set up our LLM Now
#Settings.llm = Ollama(model="llama2", keep_alive=-1)
Settings.llm = Ollama(model="llama2", request_timeout=1000)

Settings.llm.base_url="http://127.0.0.1:8080"

logger.info("INFO: Running the Bootstrap")
## Let us setup Vectorindex for Vectorstoreindex
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

#make sure that you ave run the site_scrapper.py Job to pull data in the raw directory 
logger.info("INFO: Reading the data")
# reading up Paul Graham
documents = SimpleDirectoryReader("~/data/paul_graham/").load_data()
# Reading up Zerodha Varsity data.
#documents = SimpleDirectoryReader("/home/cdsw/data/raw/").load_data()
db = chromadb.PersistentClient(path="./chroma_db")
#db.reset()
chroma_collection = db.get_or_create_collection("quickstart-paulgraham")


logger.info("INFO:Setting up the vector Store")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")
print(response)

#initialize feedback function
from trulens_eval import Tru
tru = Tru()
tru.reset_database()

#from trulens_eval.feedback.provider import OpenAI
from trulens_eval.feedback.provider.litellm import LiteLLM #we replace litellm as a local provider

from trulens_eval import Feedback, Tru, Select
import numpy as np

# Initialize provider class, replacing OpenAI will selfhosted llm
#provider = OpenAI()

provider = LiteLLM()
provider.model_engine ="ollama/llama2"

completion_args = {"api_base": "http://127.0.0.1:8080"}
provider.completion_args = completion_args
# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App
context = App.select_context(query_engine)

from trulens_eval.feedback import Groundedness
groundedness_provider=LiteLLM()
groundedness_provider.model_engine ="ollama/llama2"
groundedness_provider.completion_args = completion_args
grounded = Groundedness(groundedness_provider=groundedness_provider)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(context.collect()) # collect context chunks into a list
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance)
    .on_input_output()
)
# Question/statement relevance between question and each context chunk.
import numpy as np
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

from trulens_eval import TruLlama
feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]

tru_query_engine_recorder = TruLlama(query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=feedbacks)

### or as context manager
#with tru_query_engine_recorder as recording:
#    query_engine.query("What did the author do growing up?")
#  
## The record of the app invocation can be retrieved from the `recording`:
#
#rec = recording.get() # use .get if only one record
## recs = recording.records # use .records if multiple
#
#display(rec)

# The results of the feedback functions can be rertireved from
# `Record.feedback_results` or using the `wait_for_feedback_result` method. The
# results if retrieved directly are `Future` instances (see
# `concurrent.futures`). You can use `as_completed` to wait until they have
# finished evaluating or use the utility method:
#
#for feedback, feedback_result in rec.wait_for_feedback_results().items():
#    print(feedback.name, feedback_result.result)

# See more about wait_for_feedback_results:
# help(rec.wait_for_feedback_results)
#records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])
#
#records.head()
#
#tru.get_leaderboard(app_ids=["LlamaIndex_App1"])

# go through a list of questions for a baseline

from trulens_eval import Tru
def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)
            
eval_questions = []
with open('./data/questions/paul_graham_eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)

display(eval_questions)
for question in eval_questions:
    with tru_query_engine_recorder as recording:
        query_engine.query(question)
        
import pandas as pd
records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])
pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]

tru.get_leaderboard(app_ids=[])



# Let us go through these guestions now for sentencewindowNode Parser
from  utils.rag_helper import *
sentence_window_index_1 = build_sentence_window_index(documents, save_dir="./data/index/sentence_index1", window_size=1)
sentence_window_engine_1 = get_sentence_window_query_engine( sentence_window_index_1)
tru_recorder_1 = get_prebuilt_trulens_recorder(
    sentence_window_engine_1,
    app_id='sentence window engine 1', feedbacks=feedbacks
)

tru.get_leaderboard(app_ids=[])


# let us go through a window size of 3
sentence_window_index_3 = build_sentence_window_index(documents, save_dir="./data/index/sentence_index3", window_size=3)
sentence_window_engine_3 = get_sentence_window_query_engine( sentence_window_index_3)
tru_recorder_3 = get_prebuilt_trulens_recorder(
    sentence_window_engine_3,
    app_id='sentence window engine 3', feedbacks=feedbacks
)
run_evals(eval_questions, tru_recorder_3, sentence_window_engine_3)
tru.get_leaderboard(app_ids=[])
