"""Home page shown when the user enters the application"""
import streamlit as st

import numpy as np
import pandas as pd

import time


TITLE = "Insights on the Forecasts"


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.title(TITLE)
