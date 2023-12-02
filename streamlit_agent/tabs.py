import streamlit as st

# Create Tabs
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

# Content for Tab 1
with tab1:
    st.header("This is Tab 1")
    st.write("Here is some text for Tab 1.")

# Content for Tab 2
with tab2:
    st.header("This is Tab 2")
    st.write("Here is some text for Tab 2.")

# Content for Tab 3
with tab3:
    st.header("This is Tab 3")
    st.write("Here is some text for Tab 3.")
