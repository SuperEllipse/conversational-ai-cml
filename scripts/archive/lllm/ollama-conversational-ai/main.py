from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
import os
import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    elements = [cl.Image(name="image1", display="inline", path="/home/cdsw/lllm/ollama-conversational-ai/assets/gemma.jpeg")]
    await cl.Message(
        content="Hello there, I am Gemma. How can I help you?", elements=elements
    ).send()
    print(os.environ["OLLAMA_HOST"])
    model = Ollama(model="gemma:2b")
    model.base_url = f'http://{os.getenv("OLLAMA_HOST")}'
    print(model.base_url)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a knowledgeable historian who answers super concisely",
            ),
            ("human", "{question}"),
        ]
    )
    print("model: ", model)
    chain = prompt | model
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):

    print("in On Message")
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")

    async for chunk in chain.astream(
        {"question": message.content},
    ):
        await msg.stream_token(chunk)

    await msg.send()


@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")
