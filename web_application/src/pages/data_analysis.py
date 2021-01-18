"""Home page shown when the user enters the application"""
import streamlit as st

import awesome_streamlit as ast
import altair as alt
from src.utils import (
    _set_graphical_settings,
    _load_variables,
    _load_store_data,
    _load_train_data,
    _display_dataframe_quickly,
    _make_line_chart,
    _make_scatter_chart
)

from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

TITLE = "Data Analysis on Sales"


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
    return promo1, promo2


@st.cache
def _sort_and_take(df, sort_col="DATE", days_to_take=365):
    return df.sort_values(sort_col)[:days_to_take]


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.title(TITLE)
        color_list = _set_graphical_settings()

        st.write("## Let's have a look at the features...")
        variables = _load_variables()
        st.code(variables)

        store_data = _load_store_data().copy()
        store_data = _preprocess_store_data(store_data)
        st.write("## Here is the store metadata")
        st.dataframe(store_data)

        train_data = _load_train_data()
        st.write("## Here is the train data for the different stores")
        _display_dataframe_quickly(train_data)

        store_1_data = train_data[train_data['Store'] == 1].sort_values(
            'Date', ascending=True).reset_index(drop=False)
        store_1_data['Date'] = pd.to_datetime(store_1_data['Date'])

        # If store open
        store_1_data_open = store_1_data[store_1_data['Open'] == 1]

        st.write("## Time series Sales/Customer views")
        st.altair_chart(
            _make_line_chart(
                store_1_data,
                x='Date',
                y='Sales',
                title="Time Serie of Sales for the Store n°1 (open+closed)"
            ),
            use_container_width=True
        )

        st.altair_chart(
            _make_line_chart(
                store_1_data_open,
                x='Date',
                y='Sales',
                title="Time Serie of Sales for the Store n°1 (open)"
            ),
            use_container_width=True
        )

        st.altair_chart(
            _make_line_chart(
                store_1_data,
                x='Date',
                y='Customers',
                title="Time Serie of Customers for the Store n°1"
            ),
            use_container_width=True
        )

        st.altair_chart(
            _make_line_chart(
                store_1_data_open,
                x='Date',
                y='Customers',
                title="Time Serie of Customers for the Store n°1 (open)"
            ),
            use_container_width=True
        )

        st.write("## Aggregations")
        st.write("## Grouping by day: Average Sales and customers")

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

        st.write("## Grouping by day & store type: Average Sales and customers")
        grouped_df_2 = _produce_aggregate_col_sales_customer(
            combined_data, 'StoreType'
        )
        grouped_df_2_plot = _sort_and_take(grouped_df_2)
        sales_store_chart = _make_line_chart(
            grouped_df_2_plot,
            x='DATE',
            y='Sales',
            title="Average Sales by store type",
            color='StoreType'
        )
        st.altair_chart(sales_store_chart, use_container_width=True)

        customers_store_chart = _make_line_chart(
            grouped_df_2_plot,
            x='DATE',
            y='Customers',
            title="Average number of customers by store type",
            color='StoreType'
        )
        st.altair_chart(customers_store_chart, use_container_width=True)

        st.write("## Grouping by day & store Assortment: Average Sales and customers")
        grouped_df_3 = _produce_aggregate_col_sales_customer(
            combined_data, 'Assortment'
        )
        grouped_df_3_plot = _sort_and_take(grouped_df_3)

        sales_assortment_chart = _make_line_chart(
            grouped_df_3_plot,
            x='DATE',
            y='Sales',
            title="Average Sales by store assortment",
            color='Assortment'
        )
        st.altair_chart(sales_assortment_chart, use_container_width=True)

        customers_assortment_chart = _make_line_chart(
            grouped_df_3_plot,
            x='DATE',
            y='Customers',
            title="Average Customers by store assortment",
            color='Assortment'
        )
        st.altair_chart(customers_assortment_chart, use_container_width=True)

        st.write("## Groupby Weeks, StoreType - Sales and Customers")

        combined_data2 = _produce_aggregate_freq_store_sales_customer(
            combined_data, "W-MON"
        )
        combined_data2_plot = _sort_and_take(
            combined_data2, 'Date', len(grouped_df_3.index)
        )
        weeks_store_sales_chart = _make_line_chart(
            combined_data2_plot,
            x='Date',
            y='Sales',
            title="Weekly Total Sales by store type",
            color='StoreType'
        )
        st.altair_chart(weeks_store_sales_chart, use_container_width=True)

        weeks_store_customers_chart = _make_line_chart(
            combined_data2_plot,
            x='Date',
            y='Customers',
            title="Weekly Total Customers by store type",
            color='StoreType'
        )
        st.altair_chart(weeks_store_customers_chart, use_container_width=True)

        st.write("## Group by Weeks, StoreType - Sales and Customers")

        combined_data3 = _produce_aggregate_freq_store_sales_customer(
            combined_data, "MS"
        )
        combined_data3_plot = _sort_and_take(
            combined_data3, 'Date', len(grouped_df_3)
        )
        months_store_sales_chart = _make_line_chart(
            combined_data3_plot,
            x='Date',
            y='Sales',
            title="Monthly Total Sales by store type",
            color='StoreType'
        )
        st.altair_chart(months_store_sales_chart, use_container_width=True)

        st.write("## Group by DayOfTheWeek & Store_types - Sales")
        combined_data_noholidays = _produce_aggregate_day_week_sales(
            combined_data, 0
        )
        combined_data_noholidays_plot = _sort_and_take(
            combined_data_noholidays, 'DayOfWeek', len(
                combined_data_noholidays)
        )
        combined_data_noholidays_chart = _make_line_chart(
            combined_data_noholidays_plot,
            x='DayOfWeek',
            y='Total Sales',
            title="Total Sales per day of week",
            color='StoreType'
        )
        st.altair_chart(combined_data_noholidays_chart,
                        use_container_width=True)

        combined_data_holidays = _produce_aggregate_day_week_sales(
            combined_data, 1
        )
        combined_data_holidays_plot = _sort_and_take(
            combined_data_holidays, 'DayOfWeek', len(combined_data_holidays)
        )
        combined_data_holidays_chart = _make_line_chart(
            combined_data_holidays_plot,
            x='DayOfWeek',
            y='Total Sales',
            title="Total Sales per day of week for holidays",
            color='StoreType'
        )
        st.altair_chart(combined_data_holidays_chart, use_container_width=True)

        st.write("## Groupby DayOfTheWeek & Assortment - Sales")
        grouped_df_6 = combined_data.groupby(['DayOfWeek', 'Assortment'])[
            'Sales'].sum().reset_index(name='Total Sales')
        grouped_df_6_plot = _sort_and_take(
            grouped_df_6, 'DayOfWeek', len(combined_data_holidays)
        )
        grouped_df_6_chart = _make_line_chart(
            grouped_df_6_plot,
            x='DayOfWeek',
            y='Total Sales',
            title="Total Sales per day of week and assortment",
            color='Assortment'
        )
        st.altair_chart(grouped_df_6_chart, use_container_width=True)

        st.write("## Groupby DayOfTheWeek - Number of stores")
        grouped_df_7 = combined_data.groupby(['DayOfWeek', 'StoreType', 'Open'])[
            'Store'].count().reset_index(name='NB_stores')
        store_types = np.unique(grouped_df_7['StoreType'])
        grouped_df_7_plot = grouped_df_7[grouped_df_7['Open'] == 1]
        for store in store_types:
            grouped_df_7_plot_current = grouped_df_7_plot[grouped_df_7_plot["StoreType"] == store]
            grouped_df_7_plot_current = _sort_and_take(
                grouped_df_7_plot_current, 'DayOfWeek', len(
                    grouped_df_7_plot_current)
            )
            grouped_df_7_chart = _make_line_chart(
                grouped_df_7_plot_current,
                x='DayOfWeek',
                y=alt.Y('NB_stores', scale=alt.Scale(zero=False)),
                title=f"Total Open Shops of for store {store}",
            )
            st.altair_chart(grouped_df_7_chart, use_container_width=True)

        st.write("## Customers versus Sales")
        # scatter_df = combined_data[combined_data['Open'] == 1]

        # scatter_df_chart = _make_scatter_chart(
        #     scatter_df,
        #     x='Customers',
        #     y='Sales',
        #     color='StoreType',
        #     title='Scatterplot of customers by sales for store colored by types'
        # )
        # st.altair_chart(scatter_df_chart, use_container_width=True)
        # scatter_df_chart = _make_scatter_chart(
        #     scatter_df,
        #     x='Customers',
        #     y='Sales',
        #     color='Promo',
        #     title='Scatterplot of customers by sales for store colored by promo'
        # )
        # st.altair_chart(scatter_df_chart, use_container_width=True)

        st.write("## Classical plots")
        promo1, promo2 = _produce_promo_sales(combined_data, 'Sales')
        promo1_chart = _make_line_chart(
            promo1,
            x='Date',
            y='Avg Sales',
            title="Impact of promos on Sales Performance",
        )
        promo2_chart = _make_line_chart(
            promo2,
            x='Date',
            y='Avg Sales',
            title="Impact of promos on Sales Performance",
        )
        st.altair_chart(promo1_chart + promo2_chart, use_container_width=True)

        # fig, ax = plt.subplots()
        # ax.plot(promo1['Date'], promo1['Avg Sales'], label='without promo')
        # ax.plot(promo2['Date'], promo2['Avg Sales'], label='with promo')
        # ax.patch.set_alpha(0)
        # ax.legend()
        # ax.set_title('Impact of promos on Sales Performance')
        # fig.show()

        # promo1 = combined_data[(combined_data['Promo'] == 0) &
        #                        (combined_data['Open'] == 1)]
        # promo2 = combined_data[(combined_data['Promo'] == 1) &
        #                        (combined_data['Open'] == 1)]
        # promo1.reset_index(level=0, inplace=True)
        # promo2.reset_index(level=0, inplace=True)
        # promo1 = promo1.groupby([pd.Grouper(key='Date', freq='W-MON')]
        #                         )['Customers'].mean().reset_index(name="Mean Customers")
        # promo2 = promo2.groupby([pd.Grouper(key='Date', freq='W-MON')]
        #                         )['Customers'].mean().reset_index(name="Mean Customers")

        # fig, ax = plt.subplots()
        # ax.plot(promo1['Date'], promo1['Mean Customers'],
        #         label='without promo')
        # ax.plot(promo2['Date'], promo2['Mean Customers'], label='with promo')
        # ax.patch.set_alpha(0)
        # ax.legend()
        # ax.set_title('Impact of promos on Customers')
        # fig.show()

        # tmp = combined_data.groupby(['CompetitionDistance'])[
        #     'Customers'].mean().reset_index(name='Avg Sales')
        # fig, ax = plt.subplots()
        # ax.plot(tmp['CompetitionDistance'], tmp['Avg Sales'])
        # ax.patch.set_alpha(0)
        # fig.show()

        # combined_data.head()

        # stores = combined_data.groupby(['StoreType'])[
        #     'Store'].nunique().reset_index(name='NB Stores')
        # stores.head()

        # fig, ax = plt.subplots()
        # ax.bar(stores['StoreType'], stores['NB Stores'], color=color_list[1])
        # ax.patch.set_alpha(0)
        # fig.show()

        # assortment = combined_data.groupby(['Assortment'])[
        #     'Store'].nunique().reset_index(name='NB Stores')
        # assortment.head()

        # fig, ax = plt.subplots()
        # ax.bar(assortment['Assortment'],
        #        assortment['NB Stores'], color=color_list[1])
        # ax.patch.set_alpha(0)
        # fig.show()
