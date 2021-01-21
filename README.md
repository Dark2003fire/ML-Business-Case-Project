# ML-Business-Case-Project
Machine Learning Business Case Project

## Description

This an open source project for the Machine Learning Business Case course 
with the [Wavestone data-lab 👨‍🔬](https://www.wavestone.com/en/). 

Our client is handling around 3 000 stores in 7 different countries and wants to forecast 
up to 6-weeks of sales given historical data and various features.

## Approach

The approach taken for modeling is quite simple, since we apply a Random Forest Regressor to the processed and encoded features of the base dataset. 🌳

## About

This app is maintained by a team of 5 contributors 💪. You can learn more and
check out the source code [on Github](https://github.com/SushiFou/ML-Business-Case-Project).

## Run web application demo

A small web application has been made with [Streamlit](https://www.streamlit.io/) to visualize the client's data and forecasts.

The application can be run locally in the `web_application` folder by running: 
```
pip install -r requirements.txt 
streamlit run main.py
```

A lighter version of the app is also available [here](https://share.streamlit.io/alexzajac/ml-business-case-project/main/web_application/main.py) 🔥
