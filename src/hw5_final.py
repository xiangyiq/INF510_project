# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:05:03 2019

@author:
"""

import pandas as pd
import numpy as np
from pandas_datareader import data
import urllib
from bs4 import BeautifulSoup
from datetime import date
from statsmodels.api import OLS
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os


# The first step: obtain data from web
def obtain_data(source, file=''):
# source: remote or local, it means the method to obtain data.
# address: the address where to get data. If source=remote, the address will be API or website url.
#                                         If source=local, the address will be local path to read data.
    save_path = os.path.dirname(os.getcwd())+'\\data'
    if source == 'remote':
    # dataset 1: obtain data of stock price from yahoo finance
        # read data from remote api
        try:
            df_apple = data.DataReader('AAPL',data_source='yahoo',start='2016-01-01',end='2019-11-27')
            # convert to pandas.DataFrame
            df_apple = pd.DataFrame(df_apple)
            # save on disk as csv
            save_apple = save_path + '\\AAPL.csv'
            df_apple.to_csv(save_apple)
        except:
            print('connection failed')

    # dataset 2: obtain data of DowJones Index from yahoo finance
        # read data from yahoo finance
        try:
            df_dji = data.DataReader('^DJI',data_source='yahoo',start='2016-01-01',end='2019-11-27')
            # convert Series to pandas.DataFrame
            df_dji = pd.DataFrame(df_dji)
            # save on disk as csv
            save_dji = save_path + '\\DJI.csv'
            df_dji.to_csv(save_dji)
        except:
            print('connection failed')

    # dataset 3: obtain data of federal funds rate from https://fred.stlouisfed.org/
        try:
            url = 'https://fred.stlouisfed.org/series/FEDFUNDS'
            import urllib.request
            with urllib.request.urlopen(url) as response:
               html = response.read()

            s=BeautifulSoup(html,'lxml')
            table = s.table
            for tr in table.find_all('a'):
                link = tr.get('href')

            source_url = 'https://fred.stlouisfed.org'+link
            a = urllib.request.urlopen(source_url)
            b = a.read()
            c = BeautifulSoup(b,'lxml')
            d = c.p
            e = str(d).split('\r\n')
            f = e[500:len(e)-1]
            g = []
            for i in f:
                x = i.split(' ')
                g.append([x[0],x[3]])
            h = pd.DataFrame(g)
            df_ffr = h
            df_ffr.columns = ['date','rate']
            # save on disk as csv
            save_ffr = save_path + '\\FFR.csv'
            df_ffr.to_csv(save_ffr,index=None)

        # dataset 4: obtain data of google from yahoo finance
            # read data from yahoo finance
            df_google = data.DataReader('GOOG',data_source='yahoo',start='2016-01-01',end='2019-11-27')
            # convert to pandas.DataFrame
            df_google = pd.DataFrame(df_google)
            # save on disk as csv
            save_google = save_path + '\\GOOGLE.csv'
            df_google.to_csv(save_google)
        except:
            print('connection failed')

    # dataset 5: obtain data of GDP
        try:
            url = 'https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=US'
            html = urllib.request.urlopen(url)
            raw_data = html.read()
            soup = BeautifulSoup(raw_data,'lxml')
            string = str(soup)
            start_point = string.find('indicatorData')
            end_point = start_point+4000
            new_string = string[start_point:end_point]
            split_list = new_string.split('^3')
            table = []
            for i in range(1,len(split_list)):
                x = split_list[i].split(',')
                table.append([x[1],x[4]])
            df_gdp = pd.DataFrame(table)
            df_gdp.columns = ['gdp_growth','year']
            # save on disk as csv
            save_gdp = save_path + '\\GDP.csv'
            df_gdp.to_csv(save_gdp,index=None)
        except:
            print('connection failed')

        return df_apple, df_dji, df_ffr, df_google, df_gdp

    if source == 'local':
        try:
            file_address = save_path + '\\' + file
            df = pd.read_csv(file_address)
        except:
            print('address error')
    return df

# obtain data from web
df_apple, df_dji, df_ffr, df_google, df_gdp = obtain_data('remote')

# The second step: process data and combine them together

# process df_apple, df_dji, df_google
df1 = pd.concat([df_apple['Adj Close'], df_dji['Adj Close'], df_google['Adj Close']],axis = 1)
df2 = df1.reset_index()
df2['year-mon'] = df2['Date'].apply(lambda x: date.strftime(x,'%Y%m'),1)
df2['year'] = df2['Date'].apply(lambda x: x.year,1)

# post-process df_ffr
df3 = df_ffr
df3['year-mon'] = df3['date'].apply(lambda x: x[0:4]+x[5:7],1)
# change the variable type
df3['rate'] = df3['rate'].astype(float)

# merge the two dataframes
df4 = pd.merge(df2,df3,on='year-mon')

# post-process df_gdp
# initial df5
df5 = df_gdp
# strip quote in variable year and convert it to int
df5['year'] = df5['year'].apply(lambda x: int(x.strip('"')))
# strip quote in variable gdp and round it.
df5['gdp_growth'] = df5['gdp_growth'].apply(lambda x: round(float(x.strip('"')),3))

# merge df4 and df5
df = pd.merge(df4,df5,on='year')
# rename columns
df.columns = ['date','apple','DJI','google','year-mon','year','date_','rate','gdp_growth']
# subset
df = df[['date','apple','DJI','google','rate','gdp_growth']]
# By now, we get the final processed dataframe.
# reset column date as index
df.set_index('date',inplace=True)

save_path = os.path.dirname(os.getcwd())+'\\data'
# output the clean data to local
save_cleaned_df = save_path + '\\cleaned_df.csv'
df.to_csv(save_cleaned_df)

# The third step: data visualization
# The first figure focuses on stock price of Apple, Google, and Dow Jones Index
fig, ax1 = plt.subplots(figsize= (10,6))

ax1.plot(df['apple'], 'r', label='$apple$')
ax1.plot(df['google'], 'b', label='$google$')
ax1.legend(loc=0)
# instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
ax2.plot(df['DJI'], 'y')
ax2.set_ylabel('Dow Jones Index',color='y')
#plt.plot(c_4.mean(axis=1), 'm', label='$f=0.5$')
plt.title('The stock price of Apple, Google, and Dow Jones Index')
plt.xlabel('date')
plt.xticks(rotation=45)

# The second figure focuses on rate and gdp growth
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df['rate'], 'b', label='$rate$')
ax1.legend(loc=0)
# instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
ax2.plot(df['gdp_growth'], 'y')
ax2.set_ylabel('gdp_growth',color='y')
#plt.plot(c_4.mean(axis=1), 'm', label='$f=0.5$')
plt.title('The Federal fund rate and gdp growth')
plt.xlabel('date')

# The fourth step: correlation and regression
# correlation matrix
df.corr()

# regression
X = df[['DJI','google','rate','gdp_growth']]
y = df['apple']
# split the dataframe into train dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# create linear regression model
model = OLS(y_train, X_train).fit()

# summary the model
model.summary()

# use the model to predict stock price of Apple and use RMSE to evaluate it.
predictions = model.predict(X_train)
# sort predictions by date
predictions = pd.DataFrame(predictions).sort_index()
# compare the actual stock price of Apple
actual_price = pd.DataFrame(y_train).sort_index()
# plot comparison
plt.figure(figsize=(10, 6))
plt.plot(actual_price, label='actual price')
plt.plot(predictions, label='predicted price')
plt.title('The comparison of actual price and predicted price')
plt.legend(loc=0)
plt.xlabel('date')

# model evaluation
mean_squared_error(actual_price, predictions)

# retest it on test data set
predictions = model.predict(X_test)
# sort predictions by date
predictions = pd.DataFrame(predictions).sort_index()
# compare the actual stock price of Apple
actual_price = pd.DataFrame(y_test).sort_index()
# plot comparison
plt.figure(figsize=(10, 6))
plt.plot(actual_price, label='actual price')
plt.plot(predictions, label='predicted price')
plt.title('The comparison of actual price and predicted price')
plt.legend(loc=0)
plt.xlabel('date')

# model evaluation
mean_squared_error(actual_price, predictions)

if __name__ == '__main__':
    obtain_data('remote')
