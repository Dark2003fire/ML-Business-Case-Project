import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import pandas as pd
from pathlib import Path
import pandas as pd
import requests
from io import StringIO, BytesIO
from joblib import load
import base64
from src.download import download_large_file_from_google_drive
import gdown


def _make_line_chart(df, x="", y="", title="", height=400, **encode_args):
    chart = alt.Chart(df, title=title)\
        .mark_line().encode(x=x, y=y, **encode_args)\
        .properties(height=height)\
        .interactive()
    return chart


def _make_scatter_chart(df, x="", y="", title="", height=400, **encode_args):
    chart = alt.Chart(df, title=title)\
        .mark_point().encode(x=x, y=y, **encode_args)\
        .properties(height=height)\
        .interactive()
    return chart


def _make_bar_chart(df, x="", y="", title="", height=400, **encode_args):
    chart = alt.Chart(df, title=title)\
        .mark_bar().encode(x=x, y=y, **encode_args)\
        .properties(height=height)\
        .interactive()
    return chart


def _streamlit_theme():
    font = "IBM Plex Mono"
    primary_color = "#F63366"
    font_color = "#262730"
    grey_color = "#f0f2f6"
    base_size = 14
    lg_font = base_size * 1.25
    sm_font = base_size * 0.8  # st.table size
    xl_font = base_size * 1.75

    config = {
        "config": {
            'mark': {'tooltip': {'content': 'encoding'}},
            "view": {"fill": grey_color},
            "arc": {"fill": primary_color},
            "area": {"fill": primary_color},
            "circle": {"fill": primary_color, "stroke": font_color, "strokeWidth": 0.5},
            "line": {"stroke": primary_color},
            "path": {"stroke": primary_color},
            "point": {"stroke": primary_color},
            "rect": {"fill": primary_color},
            "shape": {"stroke": primary_color},
            "symbol": {"fill": primary_color},
            "title": {
                "font": font,
                "color": font_color,
                "fontSize": lg_font,
                "anchor": "middle",
            },
            "axis": {
                "titleFont": font,
                "titleColor": font_color,
                "titleFontSize": sm_font,
                "labelFont": font,
                "labelColor": font_color,
                "labelFontSize": sm_font,
                "grid": True,
                "gridColor": "#fff",
                "gridOpacity": 1,
                "domain": False,
                # "domainColor": font_color,
                "tickColor": font_color,
            },
            "header": {
                "labelFont": font,
                "titleFont": font,
                "labelFontSize": base_size,
                "titleFontSize": base_size,
            },
            "legend": {
                "titleFont": font,
                "titleColor": font_color,
                "titleFontSize": sm_font,
                "labelFont": font,
                "labelColor": font_color,
                "labelFontSize": sm_font,
            },
            "range": {
                "category": ["#f63366", "#fffd80", "#0068c9", "#ff2b2b", "#09ab3b"],
                "diverging": [
                    "#850018",
                    "#cd1549",
                    "#f6618d",
                    "#fbafc4",
                    "#f5f5f5",
                    "#93c5fe",
                    "#5091e6",
                    "#1d5ebd",
                    "#002f84",
                ],
                "heatmap": [
                    "#ffb5d4",
                    "#ff97b8",
                    "#ff7499",
                    "#fc4c78",
                    "#ec245f",
                    "#d2004b",
                    "#b10034",
                    "#91001f",
                    "#720008",
                ],
                "ramp": [
                    "#ffb5d4",
                    "#ff97b8",
                    "#ff7499",
                    "#fc4c78",
                    "#ec245f",
                    "#d2004b",
                    "#b10034",
                    "#91001f",
                    "#720008",
                ],
                "ordinal": [
                    "#ffb5d4",
                    "#ff97b8",
                    "#ff7499",
                    "#fc4c78",
                    "#ec245f",
                    "#d2004b",
                    "#b10034",
                    "#91001f",
                    "#720008",
                ],
            },
        }
    }
    return config


def _get_table_download_link(df, store_id):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}">Download sales forecast of store {store_id} as CSV...</a>'


@st.cache
def _load_variables(
    file_id="10p7JyO2DNkWbMRZoMNVPmipy1msZpBEV"
):
    dwn_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return requests.get(dwn_url).text


@st.cache
def _load_bytes(dwn_url):
    response = requests.get(dwn_url)
    content = response.content
    return BytesIO(content)


def _load_image(
    file_id="1_YJ_9jrJnCKvmawe-gVGuhZ9BaMgCYkt"
):
    dwn_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return _load_bytes(dwn_url)


@st.cache(allow_output_mutation=True)
def _load_model(
    file_id="1PyqEsxgEyyLcBOzMd7UhPpFKudRGT2A-"
):
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    dwn_url = f"https://drive.google.com/uc?id={file_id}"

    model_weights_path = Path("model/model.joblib")
    if not model_weights_path.exists():
        with st.spinner("Downloading model... this may take awhile! \n"):
            gdown.download(dwn_url, model_weights_path.as_posix())

    model = load(model_weights_path)
    return model


@st.cache(allow_output_mutation=True)
def _load_forecast_model(
    file_id="1yPK9wAUgfkYm_FpDASOmqW2oac8Dk_d5"
):
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    dwn_url = f"https://drive.google.com/uc?id={file_id}"

    model_weights_path = Path("model/model_forecast.joblib")
    if not model_weights_path.exists():
        with st.spinner("Downloading forecast model... this may take awhile! \n"):
            gdown.download(dwn_url, model_weights_path.as_posix())

    model = load(model_weights_path)
    return model


@st.cache
def _load_dataframe(file_id, **read_kwargs):
    dwn_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    text = requests.get(dwn_url).text
    csv_raw = StringIO(text)
    return pd.read_csv(csv_raw, **read_kwargs)


def _load_train_data_analysis(
    file_id="1kx5sSTcRj4aVS8KZgSCcdo9-5i1axh5n",
    **read_kwargs,
):
    return _load_dataframe(file_id, **read_kwargs)


def _load_test_data_analysis(
    file_id="17ur-ILBNAZDgjpqgPU1XBLYSIXc5cn5d",
    **read_kwargs,
):
    return _load_dataframe(file_id, **read_kwargs)


def _load_train_data(
    file_id="1IBTlAoYKpX64r8sVDNs9yfF3FGWFxu2k",
    **read_kwargs,
):
    return _load_dataframe(file_id, **read_kwargs)


def _load_test_data(
    file_id="1b05eSnGQxrfFLywxBF37z00H1PIWOIMU",
    **read_kwargs,
):
    return _load_dataframe(file_id, **read_kwargs)


def _load_store_data(
    file_id="1IHr_vKHZ0P0lUIAksJ9joRLUoUtZdDSY",
):
    return _load_dataframe(file_id)


def _display_dataframe_quickly(df, min_rows=5, max_rows=1000, **st_dataframe_kwargs):
    """Display a subset of a DataFrame or Numpy Array to speed up app renders.

    Parameters
    ----------
    df : DataFrame | ndarray
        The DataFrame or NumpyArray to render.
    max_rows : int
        The number of rows to display.
    st_dataframe_kwargs : Dict[Any, Any]
        Keyword arguments to the st.dataframe method.
    """
    n_rows = len(df)
    if n_rows <= max_rows:
        # As a special case, display small dataframe directly.
        st.write(df)
    else:
        # Slice the DataFrame to display less information.
        end_row = st.slider('Rows to display', min_rows, max_rows)
        df = df[:end_row]

        # Display everything.
        st.dataframe(df, **st_dataframe_kwargs)
        st.text(f'Displaying rows 0 to {end_row - 1} of {n_rows}.')


def _override_color_styles(color="#50307B"):
    """Compile some hacky CSS to override the theme color."""
    st.markdown(
        """
        <style>
        .reportview-container .markdown-text-container {
            font-family: monospace;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        .Widget>label {
            color: white;
            font-family: monospace;
        }
        [class^="st-b"]  {
            color: white;
            font-family: monospace;
        }
        .st-bb {
            background-color: transparent;
        }
        .st-at {
            background-color: #50307B;
        }
        footer {
            font-family: monospace;
        }
        .reportview-container .main footer, .reportview-container .main footer a {
            color: #50307B;
        }
        header .decoration {
            background-image: none;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
