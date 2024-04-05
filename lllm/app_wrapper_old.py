import subprocess
#~/lllm/ollama-conversational-ai/main.py
os.environ["OLLAMA_HOST"]="127.0.0.1:8080"
import os
subprocess.run(" chainlit run  --host localhost --port $CDSW_READONLY_PORT ~test.py ",shell=True, env=dict(os.environ))