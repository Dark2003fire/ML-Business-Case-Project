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


def _get_forecast_of_shops(ID_shops, y_train, y_test, forecasts):
    charts, dfs = [], []
    for ID_shop in ID_shops:
        data_init = y_train[y_train['Store'] == ID_shop][:50]
        data_init.rename(columns={"Sales": 'Past Sales'}, inplace=True)
        data_init2 = y_test[y_test['Store'] == ID_shop]
        data = forecasts[forecasts['Store'] == ID_shop]
        dfs.append((ID_shop, data))
        test2 = pd.concat([data_init['Past Sales'], data_init2['Sales']]).reset_index(
            name='Past Sales')
        test2.set_index('Date', inplace=True)
        test2.sort_index(ascending=True, inplace=True)
        link = pd.DataFrame({
            'Date': pd.to_datetime("2015-08-01"),
            'Store': ID_shop,
            'Past Sales': data.loc["2015-08-01", 'Pred']}
        )
        link.set_index('Date', inplace=True)
        link = link.astype(int)
        test2 = pd.concat([link, test2])
        test2.sort_index(ascending=True, inplace=True)
        data = pd.concat([test2, data])
        data.reset_index(inplace=True)
        data.rename(columns={"Pred": "Forecast of Sales"}, inplace=True,)
        data_plot = data.melt(
            id_vars=["Date"],
            value_vars=["Forecast of Sales", "Past Sales"]
        )

        sales_chart = _make_line_chart(
            data_plot,
            x='Date',
            y='value',
            title=f"Forecast of sales for store n°{ID_shop}",
            color=alt.Color(
                'variable',
                legend=alt.Legend(title="Sales value types"),
                scale=alt.Scale(scheme='set1')
            ),
            strokeDash='variable',
        )
        charts.append(sales_chart)
    return charts, dfs


def _evaluate_forecast_of_shops(ID_shops, test, y_test):
    charts = []
    for ID_shop in ID_shops:
        data_init = test[test['Store'] == ID_shop][:50]
        data_init.rename(columns={"Sales": 'Past Sales'}, inplace=True)
        data = y_test[y_test['Store'] == ID_shop]
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
            title=f"Evaluation of the predictions for store n°{ID_shop}",
            color=alt.Color(
                'variable',
                legend=alt.Legend(title="Sales value types"),
                scale=alt.Scale(scheme='set1')
            )
        )
        charts.append(sales_chart)
    return charts


def _get_model_predictions(
    model,
    X_test_encoded,
    y_train,
    y_test,
    should_reset_index=True,
    spinner_msg='Forecasting the next 6 weeks...'
):
    with st.spinner(spinner_msg):
        y_pred = model.predict(X_test_encoded)

    y_test['Pred'] = y_pred
    if should_reset_index:
        y_test.reset_index(inplace=True)
    y_test['Date'] = pd.to_datetime(y_test['Date'])
    y_test.set_index('Date', inplace=True)

    if y_train is not None:
        test = y_train.reset_index(level=0)
        return test, y_test
    return None, y_test


@st.cache(allow_output_mutation=True)
def _label_encoding(X):
    encoder = LabelEncoder()
    X['Assortment'] = encoder.fit_transform(X['Assortment'])
    return X


@st.cache(allow_output_mutation=True)
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
    model = RandomForestRegressor(**model_kwargs)
    with st.spinner('Training the model...'):
        model.fit(X_train_encoded, y_train)
    return model


@ st.cache
def _get_permutation_importance(model, X_test_encoded, y_test):
    return permutation_importance(model, X_test_encoded, y_test)


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.title(TITLE)

        train_data = _load_train_data(low_memory=False, index_col=0)
        X_test_forecast = _load_test_data(index_col=0)
        combined_data = train_data
        combined_data_forecast = X_test_forecast.copy()
        model_data = combined_data.drop(columns=['PromoInterval', 'Customers'])
        combined_data_forecast.drop(
            columns=['PromoInterval', 'CD_zscore'], inplace=True)

        # X, y separation
        y = model_data[['Date', 'Store', 'Sales']]
        X = model_data.drop(columns='Sales')
        X['Date'] = pd.to_datetime(X['Date'])
        y['Date'] = pd.to_datetime(y['Date'])

        # Data for evaluating performance
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

        X_train_encoded = _one_hot_encoding(X_train)
        X_train_encoded = _label_encoding(X_train_encoded)
        X_test_encoded = _one_hot_encoding(X_test)
        X_test_encoded = _label_encoding(X_test_encoded)
        X_test_forecast_encoded = _one_hot_encoding(
            combined_data_forecast)
        X_test_forecast_encoded = _label_encoding(
            X_test_forecast_encoded)
        X_test_forecast_encoded.fillna(-1, inplace=True)
        forecasts = X_test_forecast_encoded[['Date', 'Store']]

        X_train_encoded.set_index(['Date'], inplace=True)
        X_test_encoded.set_index(['Date'], inplace=True)
        y_train.set_index(['Date'], inplace=True)
        y_test.set_index(['Date'], inplace=True)
        X_test_forecast_encoded.set_index(['Date'], inplace=True)

        # Create encoded columns in the test sets that are missing.
        for column in np.asarray(X_train_encoded.columns):
            if column not in np.asarray(X_test_forecast_encoded.columns):
                X_test_forecast_encoded[column] = np.zeros(
                    X_test_forecast_encoded.shape[0])
            if column not in np.asarray(X_test_encoded.columns):
                X_test_encoded[column] = np.zeros(X_test_encoded.shape[0])

        # Data for forecasting
        X_final = pd.concat([X_test_encoded, X_train_encoded])
        y_final = pd.concat([y_test, y_train])

        # load models
        # model = _train_model(X_train_encoded, y_train["Sales"],
        #                      n_estimators=15)
        # st.success('First model trained successfully!')
        model_forecast = _train_model(X_final, y_final["Sales"],
                                      n_estimators=15, n_jobs=-1)
        st.success('Model trained successfully!')

        test, y_test = _get_model_predictions(
            model_forecast, X_test_encoded, y_train, y_test, spinner_msg="Evaluating predictions..."
        )
        _, y_test_forecast = _get_model_predictions(
            model_forecast, X_test_forecast_encoded, None, forecasts, False
        )
        stores = np.unique(test["Store"])
        store_choices = st.multiselect(
            "Select your stores for the 6-weeks forecast",
            list(stores),
            [64, 129, 930]
        )
        test_action = st.button('Forecast the next 6 weeks')
        if test_action:
            charts = _evaluate_forecast_of_shops(store_choices, test, y_test)
            charts_forecast, dfs_forecast = _get_forecast_of_shops(
                store_choices, y_train, y_test, y_test_forecast
            )
            for (chart, chart_forecast), (id_shop, df) in zip(
                zip(charts, charts_forecast), dfs_forecast
            ):
                st.text("\n")
                st.write(f"# Report for store n°{id_shop}")
                st.altair_chart(chart_forecast, use_container_width=True)
                df.rename(columns={"Pred": "Sales Forecast"}, inplace=True)
                st.write("## View as Tabular data")
                st.dataframe(df)
                st.markdown(
                    _get_table_download_link(df, id_shop), unsafe_allow_html=True
                )
                st.text("\n")
                st.write("## How well is the model performing on ground truth")
                st.altair_chart(chart, use_container_width=True)

            # st.write("# What is important for your sales...")
            # st.text("\n")
            # perm_importance = _get_permutation_importance(
            #     model_forecast, X_test_encoded, y_test['Sales']
            # )
            # sorted_idx = perm_importance.importances_mean.argsort()
            # bar_df = pd.DataFrame({
            #     "Features": [X_test_encoded.columns[i] for i in sorted_idx],
            #     "Permutation Importance": [perm_importance.importances_mean[i] for i in sorted_idx]
            # })
            # perm_bar_chart = _make_bar_chart(
            #     bar_df,
            #     x="Permutation Importance:Q",
            #     y=alt.Y('Features', sort='-x'),
            #     title="Most important variables predicting your sales..."
            # )
            # st.altair_chart(perm_bar_chart, use_container_width=True)
