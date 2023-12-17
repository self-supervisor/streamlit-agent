import streamlit as st
from summarising import generate_map_reduce_summary, gpt_medical_advice
from io import StringIO
from utils import format_to_markdown


st.set_page_config(page_title="Nora 🐈 for physicians 👩‍⚕️", page_icon="🐈‍")
st.title("Nora 🐈 for physicians 👩‍⚕️")


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

patient_profile_uploaded_file = st.file_uploader("Upload a patient profile txt file")

if patient_profile_uploaded_file is not None:
    stringio = StringIO(patient_profile_uploaded_file.getvalue().decode("utf-8"))
    elder_profile = stringio.read()

conversation_string = st.file_uploader("Upload a conversation txt file")

if conversation_string is not None:
    stringio = StringIO(conversation_string.getvalue().decode("utf-8"))
    conversation_string = stringio.read()

if conversation_string is not None and patient_profile_uploaded_file is not None:
    symptoms, general_mood = generate_map_reduce_summary(
        conversation_str=conversation_string, openai_api_key=openai_api_key
    )
    elder_profile = format_to_markdown(elder_profile)

    medical_advice = gpt_medical_advice(
        elder_profile, symptoms, general_mood, openai_api_key
    )
    with st.expander("Background"):
        st.write(elder_profile)

    with st.expander("Symptoms"):
        st.write(symptoms)

    with st.expander("Summary"):
        st.write(general_mood)

    with st.expander("GPT Medical Suggestions"):
        st.write(medical_advice)
