import os
import sys
import time
import subprocess
import psutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage, Settings
from IPython.display import Markdown, display
import chromadb
import ollama
import pandas as pd


#Evaluation Specific imports
from trulens_eval import TruLlama
from trulens_eval import Tru, Feedback, Select
from trulens_eval.feedback.provider.litellm import LiteLLM
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness
# we initialise the Tru object as a global object
tru = Tru()
tru.reset_database()


# Adding project utilities 
sys.path.append('/home/cdsw/utils')
import imp
import logging_config
from logging_config import get_logger
from  utils.rag_helper import *
logger = get_logger(__name__)



def check_ollama_installed():
    """
    Validates if 'ollama' is installed on the system and exits if not.
    """
    
    try:
        result = subprocess.run(['ollama', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 and "/bin/bash: ollama: command not found" in result.stderr:
            raise Exception("The 'ollama' command is not installed or not found in the system PATH.")
        print(result.stdout)
        logger.info("INFO : ollama exists")
    except Exception as e:
        print(f"Error: {e}")
        print (f"You need to have installed ollama to make this application work. Have you chosen the right runtime ?")
        sys.exit(1)  # Exit with a non-zero status code
      

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

def start_ollama_server():
    """
    Starts the Ollama server if it is not already running.
    """

    # Check if the 'ollama' process is already running
    serve_process=None
    if is_process_running('ollama'):
      logger.info(f"'INFO: ollama' is already running in background.")
      return None

    try:
      # Start the 'ollama' process in the background
      serve_process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      logger.info("INFO: Ollama server started in the bacground")
      # Wait for 2 seconds to allow the background process to start
      time.sleep(2)      

    except Exception as e:
      print(f"Error: {e}")
      # Terminate the background process if it's still running
      logger.error(f"Error starting Ollama server: {e}")
      if serve_process and serve_process.poll() is None:
          serve_process.terminate()
      sys.exit(1)
      

def initialize_ollama_settings():
    """
    Initializes and configures settings for the Ollama instance.
    """
    Settings.llm = Ollama(model="llama2", request_timeout=10000)
    Settings.llm.base_url = "http://127.0.0.1:8080"
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
  

def pull_and_test_model():
    """
    Pulls the Llama2 model from the library and tests it by sending a sample query.
    """
    start_time = time.time()
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


def setup_vector_index():
    """
    Sets up the vector store and index for querying.
    IMPORTANT: Make sure the bootstrap is run first. 
    """
    db2 = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db2.get_or_create_collection("quickstart-ollama")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("Loading the Index")
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

def execute_query(query_engine, query):
    """
    Executes a query using the specified query engine and prints the response.
    """
    response = query_engine.query(query)
    print(response)

def load_documents(directory):
    """
    Load documents from the specified directory.
    """
    return SimpleDirectoryReader(directory).load_data()
  
  
def setup_feedback_system(query_engine):
    """
    Sets up feedback mechanisms using Trulens and LiteLLM providers.
    
    Inputs:
    
    query_engine( Llamaindex query_engine Interface): Takes the query engine to be used for setting up feedbacks
    app(string): Name of the LLM Application. This will be later used leader board
    
    Returns:
    Feedbackhooks
    
    """

    
    provider = LiteLLM()
    provider.model_engine = "ollama/llama2"
    provider.completion_args = {"api_base": "http://127.0.0.1:8080"}
    
    context = App.select_context(query_engine)
    groundedness_provider = LiteLLM()
    groundedness_provider.model_engine = "ollama/llama2"
    groundedness_provider.completion_args = {"api_base": "http://127.0.0.1:8080"}
    
    grounded = Groundedness(groundedness_provider=groundedness_provider)
    
    # Define feedback functions
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on(context.collect())  # collect context chunks into a list
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )
    
    f_answer_relevance = (
        Feedback(provider.relevance)
        .on_input_output()
    )
    
    f_context_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )
    
    feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance]
    return feedbacks

def read_questions_from_file(filename):
    """
    Reads questions from a file and returns them as a list.
    """
    questions = []
    with open(filename, 'r') as file:
        for line in file:
            questions.append(line.strip())
    return questions
  
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

def run_evals(eval_questions, tru_recorder, query_engine):
    """
    Runs evaluation based on a set of questions against a type of query engine
    Inputs:
    eval_questions(list) : List of evaluation questions
    tru_recorder(Trulens Recorder object)
    query_engine(Lllama Index Query Engine Interface)
    
    Returns:
      Nothing
    """
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)

def get_records_and_tru_feedback(app="Baseline"):
    """
    Queries the Tru() database to get the feedback and the original records used for evaluation of the RAG Application
    """
    records, feedback = tru.get_records_and_feedback(app_ids=[app])
    pd.set_option("display.max_colwidth", None)
  #  records[["input", "output"] + feedback]
    return records[["input", "output"] + feedback]


def evaluate(query_engine,tru_query_engine_recorder,app,  questions):
    """
    Evaluate the RAG Application performance performance
    Parameters:
    query_engine: Llama index query_engine engine interface
    app (string ): the application name that you need to evaluate. This is stored in the leaderboard for reference
    filename: The filename including the full path containing the list of questions

    """
    eval_questions= read_questions_from_file(filename)
    display(eval_questions)
    feedbacks = setup_feedback(query_engine, app)

    for question in eval_questions:
        with tru_query_engine_recorder as recording:
            query_engine.query(question)


def run_rag_evaluations():
    """
    We run 3 different type of Indexes and use our RAG Triad to measure performance against different ways of Prompting the LLMs
    Baseline: Run against a Simple llama Index VectorIndex uses Simple sentence parser
    SentenceWindow: use window size to enhance the prompt context sent to LLM
    Auto Merge: Merge Different node sizes for enhanced prompting

    """
  
    #load the documents to be used by the RAG
    docs_path= '/home/cdsw/data/raw/'
    documents =load_documents(directory=docs_path)
    
    
    # Let us start with Baseline evalations
      # The evaluation data we will use    
    questions_file = './data/questions/evaluation_questions.txt'
    eval_questions = read_questions_from_file(questions_file)
    display(eval_questions)

    #Run Baseline
    baseline_query_engine = setup_vector_index().as_query_engine()  

    #first setup the feedback systems
    feedbacks = setup_feedback_system(baseline_query_engine)
    
    tru_recorder_baseline = get_prebuilt_trulens_recorder(
                              baseline_query_engine,
                              app_id ="Baseline", feedbacks=feedbacks)

    run_evals(eval_questions, tru_recorder_baseline, baseline_query_engine)
    display(get_records_and_tru_feedback("Baseline"))
    tru.get_leaderboard(app_ids=[])   

    
    #Run Sentence Window Indexer of window size 1    
    # Let us go through these guestions now for sentencewindowNode Parser
    sentence_window_index_1 = build_sentence_window_index(documents, save_dir="./data/index/sentence_index1", window_size=1)
    sentence_window_engine_1 = get_sentence_window_query_engine( sentence_window_index_1)
    tru_recorder_baseline = get_prebuilt_trulens_recorder(
                              sentence_window_engine_1,
                              app_id ='sentence window engine 1', feedbacks=feedbacks)
#    tru_recorder_1 = get_prebuilt_trulens_recorder( 
#                      sentence_window_engine_1,
#                      app_id='sentence window engine 1',
#                      feedbacks=feedbacks,
#    )
    run_evals(eval_questions, tru_recorder_baseline, sentence_window_engine_1)
    tru.get_leaderboard(app_ids=[])
    
    # let us go through a window size of 3
    sentence_window_index_3 = build_sentence_window_index(documents, save_dir="./data/index/sentence_index3", window_size=3)
    sentence_window_engine_3 = get_sentence_window_query_engine( sentence_window_index_3)
    tru_recorder_3 = get_prebuilt_trulens_recorder(
                              sentence_window_engine_3,
                              app_id ='sentence window engine 3', feedbacks=feedbacks)
    
    run_evals(eval_questions, tru_recorder_3, sentence_window_engine_3)
    tru.get_leaderboard(app_ids=[])

    auto_merging_index_0 = build_automerging_index(documents,save_dir="./data/index/merging_index_0", chunk_sizes=[2048,512],
    )

    auto_merging_engine_0 = get_automerging_query_engine(
        auto_merging_index_0,
        similarity_top_k=12,
        rerank_top_n=6,
    )


    tru_recorder_AM_1 = get_prebuilt_trulens_recorder(
        auto_merging_engine_0,
        app_id ='Merging Index 2048 & 512', feedbacks=feedbacks
    )

    run_evals(eval_questions, tru_recorder_AM_1, auto_merging_engine_0)

    # run Evals
    tru.get_leaderboard(app_ids=[])


    #Now using 3 Layers of Automerging
    auto_merging_index_1 = build_automerging_index(
        documents,
        save_dir="./data/index/merging_index_1",
        chunk_sizes=[2048,512,128],
    )

    auto_merging_engine_1 = get_automerging_query_engine(
        auto_merging_index_1,
        similarity_top_k=12,
        rerank_top_n=6,
    )

    tru_recorder_AM_2 = get_prebuilt_trulens_recorder(
        auto_merging_engine_1,
        app_id ='Merging Index 2048 & 512 & 128', feedbacks=feedbacks
    )

    run_evals(eval_questions, tru_recorder_AM_2, auto_merging_engine_1)

    tru.get_leaderboard(app_ids=[])    
  
  
def  main():

    
    check_ollama_installed()
    serve_process = start_ollama_server()
    initialize_ollama_settings()
    pull_and_test_model()
    

    if serve_process and serve_process.poll() is None:
        serve_process.terminate()  # Clean up the process if it's still running
    

    
    # We check here if ollama is set up and it executes fine with basic generation
    query_engine = setup_vector_index().as_query_engine()    
    execute_query(query_engine, "What is SEBI and what are its responsibilities?")

    # Let us run our Rag Evaluations
    run_rag_evaluations()
 
    
if __name__ == "__main__":
    main()



