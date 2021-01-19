"""This page is for searching and viewing the list of awesome resources"""
import logging

import streamlit as st

TITLE = "Modeling and Forecasting the Sales"


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.title(TITLE)
