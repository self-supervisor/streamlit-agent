from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import (
    VectorStoreRetrieverMemory,
    ConversationBufferMemory,
    CombinedMemory,
)
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import streamlit as st
import os
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS


def load_memory(file_path, memory_object):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            chat_input, chat_output = None, None
            for line in file:
                if line.startswith("AI Assistant:"):
                    # Check if there is a previous human input that hasn't been paired yet
                    if chat_input is not None and chat_output is not None:
                        memory_object.save_context(
                            {"input": chat_input}, {"output": chat_output}
                        )
                        chat_output = None  # Reset chat_output for the new pair

                    chat_input = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("Patient:"):
                    chat_output = line.split(":", 1)[1].strip().strip('"')

                    # Save the pair of input and output to memory
                    if chat_input is not None:
                        memory_object.save_context(
                            {"input": chat_input}, {"output": chat_output}
                        )
                        chat_input, chat_output = None, None  # Reset for the next pair

            # Handle any remaining pair at the end of the file
            if chat_input is not None and chat_output is not None:
                memory_object.save_context(
                    {"input": chat_input}, {"output": chat_output}
                )

    return memory_object


def load_profile_into_memory(file_path, memory_object):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                if ":" in line:
                    line_split = line.split(":", 1)
                    input = line_split[0]
                    output = line_split[1]
                    memory_object.save_context({"input": input}, {"output": output})
    return memory_object


st.set_page_config(page_title="Nora, a companion for elderly people", page_icon="🐈‍")
st.title("Nora 🐈 a companion for elderly people")

"""
I'm Nora 🐈 an AI Companion for your patients. Ask me about their current health.
"""

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
view_messages = st.expander("View the message contents in session state")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

# In actual usage, you would set `k` to be a higher value, but we use k=1 to show that
# the vector lookup still returns the semantically relevant information
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
vector_memory = VectorStoreRetrieverMemory(retriever=retriever)
vector_memory = load_memory("streamlit_agent/elder_conversation.txt", vector_memory)
vector_memory = load_profile_into_memory(
    "streamlit_agent/elder_profile.txt", vector_memory
)
chat_memory = ConversationBufferMemory(
    chat_memory=msgs, memory_key="chat_history_lines"
)
memory = CombinedMemory(memories=[vector_memory, chat_memory])

llm = OpenAI(openai_api_key=openai_api_key, temperature=0)  # Can be any valid LLM
_DEFAULT_TEMPLATE = """
You are an AI assistant that does two things for elderly people:

1. listen to how the person is feeling in terms of their health and note down their symptoms (without making a diagnoses or suggesting treatment)

2. be the persons friend, act as a sympathetic ear, reminisce with person about their past, discuss how their grandchildren are doing.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
llm_chain = ConversationChain(llm=llm, prompt=PROMPT, memory=memory, verbose=True,)

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
