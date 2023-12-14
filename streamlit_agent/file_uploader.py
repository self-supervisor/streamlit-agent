import streamlit as st
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()

    print(string_data)
