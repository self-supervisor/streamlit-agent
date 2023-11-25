from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st

st.set_page_config(page_title="Nora", page_icon="🤰🏻")
st.title("Nora")

# CSS to inject custom font styles
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Roboto', sans-serif;
}

h1 {
    font-weight: 700;
}

</style>
"""

# Inject custom CSS with the markdown function
st.markdown(custom_css, unsafe_allow_html=True)

"""
I'm Nora, your AI Doula. I'm all about giving you the info, support, and a listening ear during your pregnancy experience.
"""
st.markdown(
    """
    <img src="streamlit_agent/QmVP6rPi3BdSYoCPqxMKscDrB1nY8J7igqqqfaWnBQasRC.gif" alt="Animated gif" style="width: 200px;"/>
""",
    unsafe_allow_html=True,
)
# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How have you been?")

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
template = """You are an AI doula called Nora, providing empathetic support for pregnant women. If the conversation is going nowhere, suggest specific topics related to pregnancy that you can help with. Do not just ask questions, make it natural, if they say something about how they are feeling then gently inquire more about what is going on with them.

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
