"""Main module for the streamlit app"""
import streamlit as st

import awesome_streamlit as ast
import altair as alt
import src.pages.data_analysis
import src.pages.modeling
import src.pages.insights
from src.utils import _override_color_styles, _streamlit_theme

ast.core.services.other.set_logging_format()

PAGES = {
    "Statistical Analysis on Sales": src.pages.data_analysis,
    "Modeling Strategy": src.pages.modeling,
    "Insights on Forecasts": src.pages.insights,
}


def init():
    """Runs init scripts (color theme and changes)"""
    alt.themes.register("streamlit", _streamlit_theme)
    alt.themes.enable("streamlit")
    # st.write(_override_color_styles(), unsafe_allow_html=True)


def main():
    """Main function of the App"""
    init()
    st.sidebar.title("Navigation Panel")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)
    st.sidebar.title("Context")
    st.sidebar.info(
        """This an open source project for the Machine Learning Business Case course 
        with the [Wavestone data-lab 👨‍🔬](https://www.wavestone.com/en/). 
        \n Our client is 
        handling around 3 000 stores in 7 different countries and wants to forecast 
        up to 6-weeks of sales given historical data and various features.
        """
    )
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is maintained by a team of 5 contributors 💪. You can learn more and
        check out the source code [on Github](https://github.com/SushiFou/ML-Business-Case-Project).
        """
    )


if __name__ == "__main__":
    main()
