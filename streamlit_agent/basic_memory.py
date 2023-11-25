from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st
import os


def load_conversations(file_path):
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith("Nora:"):
                    message = line.split(":", 1)[1].strip().strip('"')
                    msgs.add_ai_message(message)
                elif line.startswith("Sarah"):
                    message = line.split(":", 1)[1].strip().strip('"')
                    msgs.add_user_message(message)
    return msgs


def truncate_history(messages, max_length=500):
    """
    Truncate the conversation history to a specified maximum token length.
    """
    total_length = 0
    truncated_messages = []
    for msg in reversed(messages):
        msg_length = len(msg.content.split())  # Estimate token count
        if total_length + msg_length > max_length:
            break
        truncated_messages.append(msg)
        total_length += msg_length
    return list(reversed(truncated_messages))


st.set_page_config(page_title="AIDoula", page_icon="🤰🏻")
st.title("Nora🤰")

"""
I'm Nora, your AI Doula. I'm all about giving you the info, support, and a listening ear during your pregnancy and beyond.
"""


# Truncate messages before passing them to the LLM chain
msgs = load_conversations("streamlit_agent/conversation_history.txt")
truncated_msgs_txt = truncate_history(msgs.messages)

# Create a new StreamlitChatMessageHistory instance and add truncated messages
truncated_msgs = StreamlitChatMessageHistory(key="truncated_langchain_messages")
for msg in truncated_msgs_txt:
    if msg.type == "human":
        truncated_msgs.add_user_message(msg.content)
    else:
        truncated_msgs.add_ai_message(msg.content)

memory = ConversationBufferMemory(chat_memory=truncated_msgs)
session_start_index = len(
    truncated_msgs.messages
)  # Index where the current session starts
view_messages = st.expander("View the message contents in session state")

# Get an OpenAI API Key before continuing
if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets.openai_api_key
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

# Set up the LLMChain, passing in memory
template = """You are an AI doula called Nora, providing empathetic support for pregnant women. If the conversation is going nowhere, suggest specific topics related to pregnancy that you can help with. Do not just ask questions, make it natural.

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
llm_chain = LLMChain(
    llm=OpenAI(openai_api_key=openai_api_key), prompt=prompt, memory=memory
)

# Render only messages from the current session
for msg in truncated_msgs.messages[session_start_index:]:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    response = llm_chain.run(prompt)
    st.chat_message("ai").write(response)


# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
