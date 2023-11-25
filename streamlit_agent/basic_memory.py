from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
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


st.set_page_config(page_title="AIDoula", page_icon="🤰🏻")
st.title("Nora🤰")

"""
I'm Nora, your AI Doula. I'm all about giving you the info, support, and a listening ear during your pregnancy and beyond.
"""

# Set up memory
msgs = load_conversations("streamlit_agent/conversation_history.txt")
memory = ConversationBufferWindowMemory(chat_memory=msgs, k=5)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How have you been?")

view_messages = st.expander("View the message contents in session state")

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

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
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
