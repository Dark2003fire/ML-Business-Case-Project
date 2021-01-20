"""Home page shown when the user enters the application"""
import streamlit as st

import altair as alt
from src.utils import (
    _load_variables,
    _load_store_data,
    _load_train_data_analysis,
    _display_dataframe_quickly,
    _make_line_chart,
    _make_scatter_chart,
    _make_bar_chart
)
from src.processing import (
    _preprocess_store_data,
    _produce_combined_data
)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

TITLE = "Data Analysis on Sales"


@st.cache
def _produce_aggregate_sales_customer(combined_data):
    grouped_df = combined_data.groupby(['Date']).agg(
        {'Sales': 'sum', 'Customers': 'sum'})
    grouped_df.reset_index(level=0, inplace=True)
    grouped_df['Date'] = pd.to_datetime(
        grouped_df['Date']) - pd.to_timedelta(7, unit='d')
    # To comment if daily graph
    grouped_df = grouped_df.groupby(
        [pd.Grouper(key=('Date'), freq='W-MON')]).agg({'Sales': 'sum', 'Customers': 'sum'})

    scaler = MinMaxScaler()
    grouped_df[['Sales', 'Customers']] = scaler.fit_transform(
        grouped_df[['Sales', 'Customers']])
    grouped_df["Date"] = grouped_df.index
    return grouped_df


@st.cache
def _produce_aggregate_col_sales_customer(combined_data, by_col):
    grouped_df = combined_data.groupby(['Date', by_col]).agg(
        {'Sales': 'mean', 'Customers': 'mean'})
    grouped_df.reset_index(level=1, inplace=True)
    grouped_df.reset_index(level=0, inplace=True)

    grouped_df = grouped_df.groupby([by_col, pd.Grouper(
        key=('Date'), freq='W-MON')]).agg({'Sales': 'mean', 'Customers': 'mean'})
    grouped_df.reset_index(level=0, inplace=True)
    grouped_df["DATE"] = grouped_df.index
    return grouped_df


@st.cache
def _produce_aggregate_freq_store_sales_customer(combined_data, freq):
    combined_data2 = combined_data.reset_index(level=0)
    combined_data2['Date'] = pd.to_datetime(combined_data2['Date'])
    grouped_df_4 = combined_data2.groupby(['StoreType', pd.Grouper(
        key='Date', freq=freq)]).agg({'Sales': 'sum', 'Customers': 'sum'}).reset_index().sort_values('Date')
    return grouped_df_4


@st.cache
def _produce_aggregate_day_week_sales(combined_data, holiday_code):
    combined_data_noholidays = combined_data[combined_data['SchoolHoliday'] == holiday_code]
    grouped_df_5 = combined_data_noholidays.groupby(['DayOfWeek', 'StoreType'])[
        'Sales'].sum().reset_index(name='Total Sales')
    return grouped_df_5


@st.cache
def _produce_promo_sales(combined_data, group_col):
    promo1 = combined_data[(combined_data['Promo'] == 0) &
                           (combined_data['Open'] == 1)]
    promo2 = combined_data[(combined_data['Promo'] == 1) &
                           (combined_data['Open'] == 1)]

    promo1.reset_index(level=0, inplace=True)
    promo2.reset_index(level=0, inplace=True)
    promo1 = promo1.groupby([pd.Grouper(key='Date', freq='W-MON')]
                            )[group_col].mean().reset_index(name="Avg Sales")
    promo2 = promo2.groupby([pd.Grouper(key='Date', freq='W-MON')]
                            )[group_col].mean().reset_index(name="Avg Sales")
    promo1["promo_type"] = "without promo"
    promo2["promo_type"] = "with promo"
    final_promo = pd.concat([promo1, promo2])
    return final_promo


@st.cache
def _produce_sampled_scatter(combined_data, sample_size=1000):
    scatter_df = combined_data[combined_data['Open'] == 1]
    scatter_df_reduced = scatter_df.groupby('StoreType').apply(
        lambda x: x.sample(sample_size)
    )
    return scatter_df_reduced


@st.cache
def _sort_and_take(df, sort_col="DATE", days_to_take=365):
    return df.sort_values(sort_col)[:days_to_take]


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.title(TITLE)

        st.write("## Let's have a look at the features...")
        variables = _load_variables()
        st.code(variables)

        store_data = _load_store_data().copy()
        store_data = _preprocess_store_data(store_data)
        st.write("## Here is the store metadata")
        st.dataframe(store_data)

        train_data = _load_train_data_analysis()
        st.write("## Here is the train data for the different stores")
        _display_dataframe_quickly(train_data)

        st.write("# Visualize the Sales and Customer for a specific store")
        unique_stores = np.unique(train_data['Store'])
        store_option = st.selectbox(
            "Select your store:",
            unique_stores
        )
        store_1_data = train_data[train_data['Store'] == store_option].sort_values(
            'Date', ascending=True).reset_index(drop=False)
        store_1_data['Date'] = pd.to_datetime(store_1_data['Date'])

        # If store open
        store_1_data_open = store_1_data[store_1_data['Open'] == 1]

        # regarder granularité
        st.write(
            f"## Time series Sales/Customer views for store {store_option}")
        st.altair_chart(
            _make_line_chart(
                store_1_data,
                x='Date',
                y='Sales',
                title=f"Time Serie of Sales for the Store n°{store_option} (open+closed)"
            ),
            use_container_width=True
        )

        st.altair_chart(
            _make_line_chart(
                store_1_data_open,
                x='Date',
                y='Sales',
                title=f"Time Serie of Sales for the Store n°{store_option} (open)"
            ),
            use_container_width=True
        )

        st.altair_chart(
            _make_line_chart(
                store_1_data,
                x='Date',
                y='Customers',
                title=f"Time Serie of Customers for the Store n°{store_option}"
            ),
            use_container_width=True
        )

        st.altair_chart(
            _make_line_chart(
                store_1_data_open,
                x='Date',
                y='Customers',
                title=f"Time Serie of Customers for the Store n°{store_option} (open)"
            ),
            use_container_width=True
        )

        st.write("# Aggregations")
        st.write(
            "## Total Sales and Total customers evolution over all stores (week aggegation)")

        combined_data = _produce_combined_data(store_data, train_data)
        group_df = _produce_aggregate_sales_customer(combined_data)
        group_df_plot = group_df.melt(id_vars=["Date"])
        sales_chart = _make_line_chart(
            group_df_plot,
            x='Date',
            y='value',
            title="Time Serie of Customers for the Store n°1 (open)",
            color=alt.Color('variable', legend=alt.Legend(title="Serie type"))
        )
        st.altair_chart(sales_chart, use_container_width=True)

        st.write("# Store type Statistics")
        grouped_df_2 = _produce_aggregate_col_sales_customer(
            combined_data, 'StoreType'
        )
        grouped_df_2_plot = _sort_and_take(grouped_df_2)
        sales_store_chart = _make_line_chart(
            grouped_df_2_plot,
            x='DATE',
            y='Sales',
            title="Average Sales by Store Type",
            color='StoreType'
        )
        st.altair_chart(sales_store_chart, use_container_width=True)

        customers_store_chart = _make_line_chart(
            grouped_df_2_plot,
            x='DATE',
            y='Customers',
            title="Average Number of Customers by Store Type",
            color='StoreType'
        )
        st.altair_chart(customers_store_chart, use_container_width=True)
        combined_data2 = _produce_aggregate_freq_store_sales_customer(
            combined_data, "W-MON"
        )
        combined_data2_plot = _sort_and_take(
            combined_data2, 'Date', len(combined_data2.index)
        )
        weeks_store_sales_chart = _make_line_chart(
            combined_data2_plot,
            x='Date',
            y='Sales',
            title="Weekly Total Sales by Store Type",
            color='StoreType'
        )
        st.altair_chart(weeks_store_sales_chart, use_container_width=True)

        weeks_store_customers_chart = _make_line_chart(
            combined_data2_plot,
            x='Date',
            y='Customers',
            title="Weekly Total Customers by Store Type",
            color='StoreType'
        )
        st.altair_chart(weeks_store_customers_chart, use_container_width=True)

        st.write("# Visualize Sales and Customers on Custom Aggregation")
        granularity_option = st.selectbox(
            "Select your aggregation:",
            ('Weekly', 'Monthly')
        )
        aggregate_code = {
            'Weekly': 'W-MON',
            'Monthly': 'MS'
        }
        combined_data3 = _produce_aggregate_freq_store_sales_customer(
            combined_data, aggregate_code[granularity_option]
        )
        combined_data3_plot = _sort_and_take(
            combined_data3, 'Date', len(combined_data3)
        )
        months_store_sales_chart = _make_line_chart(
            combined_data3_plot,
            x='Date',
            y='Sales',
            title=f"Total Sales by store type, aggregated {granularity_option}",
            color='StoreType'
        )
        st.altair_chart(months_store_sales_chart, use_container_width=True)
        months_store_sales_chart = _make_line_chart(
            combined_data3_plot,
            x='Date',
            y='Customers',
            title=f"Total Customer by store type, aggregated {granularity_option}",
            color='StoreType'
        )
        st.altair_chart(months_store_sales_chart, use_container_width=True)

        st.write("# Store Assortment Statistics")
        grouped_df_3 = _produce_aggregate_col_sales_customer(
            combined_data, 'Assortment'
        )
        grouped_df_3_plot = _sort_and_take(grouped_df_3)

        sales_assortment_chart = _make_line_chart(
            grouped_df_3_plot,
            x='DATE',
            y='Sales',
            title="Average Sales by Store Assortment",
            color='Assortment'
        )
        st.altair_chart(sales_assortment_chart, use_container_width=True)

        customers_assortment_chart = _make_line_chart(
            grouped_df_3_plot,
            x='DATE',
            y='Customers',
            title="Average Customers by Store Assortment",
            color='Assortment'
        )
        st.altair_chart(customers_assortment_chart, use_container_width=True)

        st.write("# Week Opening Statistics")
        holiday_option = st.selectbox(
            "Type of period",
            ('Holiday', 'Non-Holiday')
        )
        holiday_code = {
            'Holiday': 1,
            'Non-Holiday': 0
        }
        combined_data_holidays = _produce_aggregate_day_week_sales(
            combined_data, holiday_code[holiday_option]
        )
        combined_data_holidays_plot = _sort_and_take(
            combined_data_holidays, 'DayOfWeek', len(
                combined_data_holidays
            )
        )
        combined_data_holidays_chart = _make_line_chart(
            combined_data_holidays_plot,
            x='DayOfWeek',
            y='Total Sales',
            title=f"Stores Open by Day of Week & Store Type for {holiday_option}",
            color='StoreType'
        )
        st.altair_chart(combined_data_holidays_chart,
                        use_container_width=True
                        )

        grouped_df_6 = combined_data.groupby(['DayOfWeek', 'Assortment'])[
            'Sales'].sum().reset_index(name='Total Sales')
        grouped_df_6_plot = _sort_and_take(
            grouped_df_6, 'DayOfWeek', len(combined_data_holidays)
        )
        grouped_df_6_chart = _make_line_chart(
            grouped_df_6_plot,
            x='DayOfWeek',
            y='Total Sales',
            title=f"Total Sales per day of week and assortment for {holiday_option}",
            color='Assortment'
        )
        st.altair_chart(grouped_df_6_chart, use_container_width=True)

        st.write("# Number of stores Open by Day of the Week")
        grouped_df_7 = combined_data.groupby(['DayOfWeek', 'StoreType', 'Open'])[
            'Store'].count().reset_index(name='NB_stores')
        store_types = np.unique(grouped_df_7['StoreType'])
        store_type_option = st.selectbox(
            "Select your store type:",
            store_types
        )
        grouped_df_7_plot = grouped_df_7[grouped_df_7['Open'] == 1]
        grouped_df_7_plot_current = grouped_df_7_plot[grouped_df_7_plot["StoreType"]
                                                      == store_type_option]
        grouped_df_7_plot_current = _sort_and_take(
            grouped_df_7_plot_current, 'DayOfWeek', len(
                grouped_df_7_plot_current)
        )
        grouped_df_7_plot_week = grouped_df_7_plot_current.copy()
        grouped_df_7_plot_week['NB_stores'] = grouped_df_7_plot_week['NB_stores'] / 52
        grouped_df_7_chart = _make_line_chart(
            grouped_df_7_plot_week,
            x='DayOfWeek',
            y=alt.Y('NB_stores', scale=alt.Scale(zero=False)),
            title=f"Total Open Shops for store {store_type_option}",
        )
        st.altair_chart(grouped_df_7_chart, use_container_width=True)

        st.write("# Customers versus Sales")
        scatter_df_reduced = _produce_sampled_scatter(combined_data)

        scatter_df_chart = _make_scatter_chart(
            scatter_df_reduced,
            x='Customers',
            y='Sales',
            color='StoreType',
            title='Scatterplot of customers by sales for store colored by types'
        )
        st.altair_chart(scatter_df_chart, use_container_width=True)
        scatter_df_chart_promo = _make_scatter_chart(
            scatter_df_reduced,
            x='Customers',
            y='Sales',
            color='Promo:N',
            title='Scatterplot of customers by sales for store colored by promo'
        )
        st.altair_chart(scatter_df_chart_promo, use_container_width=True)

        st.write("# Classical plots")
        final_promo = _produce_promo_sales(combined_data, 'Sales')
        promo1_chart = _make_line_chart(
            final_promo,
            x='Date',
            y='Avg Sales',
            color='promo_type',
            title="Impact of promos on Sales Performance",
        )
        st.altair_chart(promo1_chart, use_container_width=True)

        st.write("# Counts of Stores Per Story type or Assortment")
        stores = combined_data.groupby(['StoreType'])[
            'Store'].nunique().reset_index(name='NB Stores')
        stores_type_chart = _make_bar_chart(
            stores,
            x='StoreType',
            y='NB Stores',
            title="Number of Stores per Store Type",
        )
        st.altair_chart(stores_type_chart, use_container_width=True)

        assortment = combined_data.groupby(['Assortment'])[
            'Store'].nunique().reset_index(name='NB Stores')
        assortment_chart = _make_bar_chart(
            assortment,
            x='Assortment',
            y='NB Stores',
            title="Number of Stores per Assortment",
        )
        st.altair_chart(assortment_chart, use_container_width=True)
