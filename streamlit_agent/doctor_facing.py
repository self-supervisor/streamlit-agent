from langchain.llms import OpenAI
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
)
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import streamlit as st
from summarising import generate_overall_summary
from utils import (
    setup_vector_db,
    load_profile_into_memory,
    generate_basic_profile_str,
    load_memory,
)


st.set_page_config(page_title="Nora 🐈 for physicians 👩‍⚕️", page_icon="🐈‍")
st.title("Nora 🐈 for physicians 👩‍⚕️")


msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Ask me anything about your patient's health.")

view_messages = st.expander("View the message contents in session state")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

vector_memory = setup_vector_db(openai_api_key)
vector_memory = load_memory("elder_conversation.txt", vector_memory)
vector_memory, line_list = load_profile_into_memory("elder_profile.txt", vector_memory)
basic_profile = generate_basic_profile_str(line_list)
st.markdown("Let's discuss how your patient is doing.\n")


chat_memory = ConversationBufferMemory(
    chat_memory=msgs, memory_key="chat_history_lines", input_key="input",
)
memory = CombinedMemory(memories=[vector_memory, chat_memory])

llm = OpenAI(openai_api_key=openai_api_key, temperature=0)  # Can be any valid LLM

_DEFAULT_TEMPLATE = """
You are talking to a doctor about their patient who you have been assisting.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Recent Conversation:
{chat_history_lines}

Current conversation:
Human: {input}"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
llm_chain = ConversationChain(llm=llm, prompt=PROMPT, memory=memory, verbose=True,)

symptoms, general_mood = generate_overall_summary(api_key=openai_api_key)
with st.expander("Background"):
    st.write(basic_profile)

with st.expander("Symptoms"):
    st.write(symptoms)

with st.expander("Summary"):
    st.write(general_mood)


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
