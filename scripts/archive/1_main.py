import subprocess
import sys
import os
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
#      ollama.pull(model)

    ollama.pull('llama2')
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