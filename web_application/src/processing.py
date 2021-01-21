import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats


def _preprocess_store_data(store_data):
    """Applies custom processing to store data"""
    store_data['CompetitionDistance'].fillna(
        store_data['CompetitionDistance'].mean(), inplace=True
    )
    store_data.fillna(-1, inplace=True)
    store_data['CD_zscore'] = np.abs(
        stats.zscore(store_data['CompetitionDistance'].values)
    )
    store_data_cleaned = store_data[store_data['CD_zscore'] < 3]
    return store_data_cleaned


@st.cache
def _produce_combined_data(store_data, train_data):
    combined_data = store_data.merge(train_data, on=['Store'])
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])
    combined_data.set_index('Date', inplace=True)
    return combined_data
