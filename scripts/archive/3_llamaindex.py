import llama_index
from llama_index.llms.ollama import Ollama
import os 

import chainlit as cl

import subprocess
import sys

import time
¸



try:
    # Start the background process
    print("I am here" )
    serve_process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for 3 seconds to allow the background process to start
    time.sleep(3)

#
#    response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
#    if ( response['done'] == True):
#      print('MODEL RESPONSE OBTAINED')
#      print(response)
      
    
#    # Terminate the background process
#    serve_process.terminate()


except Exception as e:
    print(f"Error: {e}")
    # Terminate the background process if it's still running
    if serve_process.poll() is None:
        serve_process.terminate()
    sys.exit(1)
    
    

llm = Ollama(model="gemma:2b", request_timeout=60.0)
llm.base_url= f'http://{os.environ["OLLAMA_HOST"]} '
resp = llm.complete("Who is Paul Graham?")
print(resp)


# chat message 

from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

# Stream Complete

response = llm.stream_complete("Who is Paul Graham?")


for r in response:
    print(r.delta, end="")
    
    
# Stream Chat

from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")