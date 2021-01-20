"""This page is for forecasting sales and wiewing model insights"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from src.utils import (
    _load_store_data,
    _load_train_data,
    _load_test_data,
    _display_dataframe_quickly,
    _load_model,
    _make_line_chart,
    _get_table_download_link,
    _make_bar_chart
)
from src.processing import (
    _preprocess_store_data, _produce_combined_data
)
import streamlit as st

TITLE = "Forecasting the Sales and Insights"


def _get_forecast_of_shops(ID_shops, test, y_test):
    charts, dfs = [], []
    for ID_shop in ID_shops:
        data_init = test[test['Store'] == ID_shop][:50].copy()
        data_init.rename(columns={"Sales": 'Past Sales'}, inplace=True)
        data = y_test[y_test['Store'] == ID_shop].copy()
        dfs.append((ID_shop, data))
        link = pd.DataFrame({
            'Date': pd.to_datetime("2015-06-20"),
            'Store': ID_shop,
            'Past Sales': data.loc["2015-06-20", 'Pred']}
        )
        link.reset_index(level=0, drop=True, inplace=True)
        data_init = pd.concat([link, data_init])
        data.reset_index(level=0, inplace=True)
        data = pd.concat([data_init, data])
        data.reset_index(inplace=True)
        data.rename(
            columns={"Sales": "Real Sales", "Pred": "Predicted Sales"},
            inplace=True,
        )
        data_plot = data.melt(
            id_vars=["Date"],
            value_vars=["Real Sales", "Predicted Sales", "Past Sales"]
        )

        sales_chart = _make_line_chart(
            data_plot,
            x='Date',
            y='value',
            title=f"Time Serie of Customers for the Store n°{ID_shop}",
            color=alt.Color('variable', legend=alt.Legend(
                title="Sales value types"
            )
            )
        )
        charts.append(sales_chart)
    return charts, dfs


def _get_model_predictions(
    model, X_test_encoded, y_train, y_test
):
    with st.spinner('Forecasting the next 6 weeks...'):
        y_pred = model.predict(X_test_encoded)

    y_test['Pred'] = y_pred
    y_test.reset_index(inplace=True)
    y_test['Date'] = pd.to_datetime(y_test['Date'])
    y_test.set_index('Date', inplace=True)

    test = y_train.reset_index(level=0)
    stores = np.unique(test["Store"])
    store_choices = st.multiselect(
        "Select your stores for the 6-weeks forecast",
        list(stores),
        [17]
    )
    return store_choices, test, y_test


@st.cache
def _label_encoding(X):
    encoder = LabelEncoder()
    X['Assortment'] = encoder.fit_transform(X['Assortment'])
    return X


@st.cache
def _one_hot_encoding(X):
    encoder = OneHotEncoder(sparse=False)
    features = ['StoreType', 'StateHoliday']
    X_encoded = pd.DataFrame(encoder.fit_transform(X[features]))
    X_encoded.columns = encoder.get_feature_names(features)
    tmp = X.drop(features, axis=1)
    X_encoded = pd.concat([tmp, X_encoded], axis=1)
    return X_encoded


@st.cache(allow_output_mutation=True)
def _train_model(X_train_encoded, y_train, **model_kwargs):
    model = RandomForestRegressor(**model_kwargs, n_jobs=-1)
    with st.spinner('Training the model...'):
        model.fit(X_train_encoded, y_train.values.ravel())
    return model


@st.cache
def _get_permutation_importance(model, X_test_encoded, y_test):
    return permutation_importance(model, X_test_encoded, y_test)


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.title(TITLE)

        train_data = _load_train_data(low_memory=False, index_col=0)
        test_data = _load_test_data(index_col=0)
        combined_data = train_data
        model_data = combined_data.drop(columns=['PromoInterval', 'Customers'])

        # X, y separation
        y = model_data[['Date', 'Store', 'Sales']].copy()
        X = model_data.drop(columns='Sales').copy()
        X['Date'] = pd.to_datetime(X['Date'])
        y['Date'] = pd.to_datetime(y['Date'])
        # Date is at position 2
        date_ref = pd.to_datetime(
            X.loc[0, "Date"]) - pd.to_timedelta(42, unit='d')
        X_train = X[X['Date'] <= date_ref]
        X_test = X[X['Date'] > date_ref]
        y_train = y[y['Date'] <= date_ref]
        y_test = y[y['Date'] > date_ref]

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        X_train_encoded = _one_hot_encoding(X_train).copy()
        X_train_encoded = _label_encoding(X_train_encoded).copy()
        X_test_encoded = _one_hot_encoding(X_test).copy()
        X_test_encoded = _label_encoding(X_test_encoded).copy()
        X_train_encoded.set_index(['Date'], inplace=True)
        X_test_encoded.set_index(['Date'], inplace=True)
        y_train.set_index(['Date'], inplace=True)
        y_test.set_index(['Date'], inplace=True)

        # Create encoded columns in the test sets that are missing.
        for column in np.asarray(X_train_encoded.columns):
            if column not in np.asarray(X_test_encoded.columns):
                X_test_encoded[column] = np.zeros(X_test_encoded.shape[0])

        # load model
        model = _load_model()
        st.success('Model loaded successfully!')

        store_choices, test, y_test = _get_model_predictions(
            model, X_test_encoded, y_train, y_test
        )
        test_action = st.button('Forecast the next 6 weeks')
        if test_action:
            charts, dfs = _get_forecast_of_shops(store_choices, test, y_test)
            for chart, (id_shop, df) in zip(charts, dfs):
                st.text("\n")
                st.write(f"## Report for store n°{id_shop}")
                df.rename(columns={"Pred": "Sales Forecast"}, inplace=True)
                st.dataframe(df)
                st.markdown(
                    _get_table_download_link(df, id_shop), unsafe_allow_html=True
                )
                st.text("\n")
                st.altair_chart(chart, use_container_width=True)

            st.write("# What is important for your sales...")
            st.text("\n")
            perm_importance = _get_permutation_importance(
                model, X_test_encoded, y_test['Sales']
            )
            sorted_idx = perm_importance.importances_mean.argsort()
            bar_df = pd.DataFrame({
                "Features": [X_test_encoded.columns[i] for i in sorted_idx],
                "Permutation Importance": [perm_importance.importances_mean[i] for i in sorted_idx]
            })
            perm_bar_chart = _make_bar_chart(
                bar_df,
                x="Permutation Importance:Q",
                y=alt.Y('Features', sort='-x'),
                title="Most important variables predicting your sales..."
            )
            st.altair_chart(perm_bar_chart, use_container_width=True)
        # model = _load_model()
        # _display_model_predictions(model, X_test_encoded, y_train, y_test)
