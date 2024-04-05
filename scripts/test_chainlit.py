
import chainlit as cl

import sys
sys.path.append('/home/cdsw/utils')

# In the asynchronous function
from logging_config import get_logger


# Get the shared logger
logger = get_logger(__name__)


# Log messages in the asynchronous function
logger.debug("This is a debug message from the async function.")
logger.info("This is an informational message from the async function.")




@cl.on_chat_start
def on_chat_start():
  logger.info("Inside: on_chat_start")
  logger.handlers[0].flush()
  cl.user_session.set("counter", 0)


@cl.on_message
async def on_message(message: cl.Message):
    logger.info("Inside: on_message")
    logger.handlers[0].flush()
    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)
    logger.info(f"Value of counter:{counter}")
    logger.handlers[0].flush()
    await cl.Message(content=f"You sent {counter} message(s)!").send()