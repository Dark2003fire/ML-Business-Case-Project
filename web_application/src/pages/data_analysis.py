"""Home page shown when the user enters the application"""
import streamlit as st

import awesome_streamlit as ast

TITLE = "Data Analysis on Sales"


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.title(TITLE)
