import subprocess
import os
chainlit_app_file = "~/lllm/ollama-conversational-ai/main.py"
#chainlit_app_file = "~/scripts/RAG.py"
print(f"Access the chainlit application here:\n https://read-only-{os.environ['CDSW_ENGINE_ID']}.{os.environ['CDSW_DOMAIN']}")
os.system(f"chainlit run --host localhost --port $CDSW_READONLY_PORT {chainlit_app_file}")


#This seems to work
#prompt = '\'{ \
#  "model": "gemma:2b", \
#  "prompt": "Why is the sky blue?" }\''
#print(prompt)
#full_prompt = f'curl http://localhost:8080/api/generate -d {prompt}'
#print(full_prompt)
#os.system(full_prompt)