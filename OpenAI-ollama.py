from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:8080/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="gemma:2b",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "When was India Independent?"},
    {"role": "assistant", "content": "India was independent in the year 1947"},
    {"role": "user", "content": "when is Independence Day celebrated in India?"}
  ]
)
print(response.choices[0].message.content)