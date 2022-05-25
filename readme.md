# WIG20 Stock Price Prediction

## Table of contents
[Description](#description)

[Technologies/Packages](#Technologies/Packages)

[Installation](#Installation)

## Description
WIG20 Stock Price Prediction. Dataset taken from [stooq.pl](https://stooq.pl/q/d/?s=wig20) from 07-01-1997 to 
23-05-2022. Dataset upload to WIG20 Stock Price Prediction project 
[here](https://github.com/maciejwaruszewski/WIG20_forecasting/blob/master/wig20_d.csv).
Dataset has been trained by 60 previous day. WIG20 Stock Price Forecast done for the next 20 days, starting 
from 24-05-2022. To assign dates for each predictive WIG20 stock price there was a new dataframe with range of date 
needed. For visualisation purposes range of date taken - last 60 days + 20 predictive days.

## Installation
$ git clone https://github.com/maciejwaruszewski/WIG20_forecasting.git && cd maciejwaruszewski

## Packages
* pandas
* numpy
* matplotlib
* sklearn.preprocessing
* keras