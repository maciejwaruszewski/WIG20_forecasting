import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Importing the dataset, sorting columns
dataset_train = pd.read_csv('wig20_d.csv')
new_cols = ['Data', 'Zamkniecie', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Wolumen']
dataset_train = dataset_train.reindex(columns=new_cols)
training_set = dataset_train.iloc[:, 1:2].values

# 'feature_range = (0,1)' - training data is scaled to have values between 0 and 1
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps (look back 60 days) and 1 output
# 'x_train' Input with 60 previous days' stock prices
X_train = []
# 'y_train' Output with next day's stock price
y_train = []
for i in range(60, 7292):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# 20% of Neurons will be ignored
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
# 'X_train' Independent variables
# 'y_train' Output Truths that we compare X_train to.
regressor.fit(X_train, y_train, epochs=1, batch_size=32)

dataset = pd.read_csv('wig20_d.csv')
dataset_test = dataset.iloc[7233:7294]
new_cols = ['Data', 'Zamkniecie', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Wolumen']
dataset_test = dataset_test.reindex(columns=new_cols)

# Getting the predicted stock price from 2022-05-24
# We need 60 previous inputs for each day. Combine 'dataset_train' and 'dataset_test'
dataset_total = pd.concat((dataset_train['Zamkniecie'], dataset_test['Zamkniecie']), axis=0)
# Extract Stock Prices for Test time period, plus 60 days previous
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Predict the Stock Price
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# New dataframes for predicted data
predicted_otwarcie = pd.DataFrame(predicted_stock_price, columns={'Otwarcie'})
predicted_najwyzszy = pd.DataFrame(predicted_stock_price, columns={'Najwyzszy'})
predicted_najnizszy = pd.DataFrame(predicted_stock_price, columns={'Najnizszy'})
predicted_zamkniecie = pd.DataFrame(predicted_stock_price, columns={'Zamkniecie'})
predicted_wolumen = pd.DataFrame(predicted_stock_price, columns={'Wolumen'})

# New dataframe with dates of prediction WIG20 price
predicted_date = pd.DataFrame({'Data': ['2022-05-24', '2022-05-25', '2022-05-26', '2022-05-27', '2022-05-28',
                                        '2022-05-29', '2022-05-30', '2022-05-31', '2022-06-01', '2022-06-02',
                                        '2022-06-03', '2022-06-04', '2022-06-05', '2022-06-06', '2022-06-07',
                                        '2022-06-08', '2022-06-09', '2022-06-10', '2022-06-11', '2022-06-12']})

wig20_forecast = pd.DataFrame()

wig20_forecast['Data'] = predicted_date
wig20_forecast['Zamkniecie'] = predicted_zamkniecie
wig20_forecast['Otwarcie'] = predicted_otwarcie
wig20_forecast['Najwyzszy'] = predicted_najwyzszy
wig20_forecast['Najnizszy'] = predicted_najnizszy
wig20_forecast['Wolumen'] = predicted_wolumen

wig20_total_date = pd.concat((dataset_train['Data'], wig20_forecast['Data']), axis=0)
wig20_total_price = pd.concat((dataset_train['Zamkniecie'], wig20_forecast['Zamkniecie']), axis=0)
wig20_total_open = pd.concat((dataset_train['Otwarcie'], wig20_forecast['Otwarcie']), axis=0)
wig20_total_max = pd.concat((dataset_train['Najwyzszy'], wig20_forecast['Najwyzszy']), axis=0)
wig20_total_min = pd.concat((dataset_train['Najnizszy'], wig20_forecast['Najnizszy'], ), axis=0)
wig20_total_volumen = pd.concat((dataset_train['Wolumen'], wig20_forecast['Wolumen']), axis=0)

# Saving range of 30 years date for visualisation purposes
wig20_total_date.to_csv('wig20_date.csv')


# Plot to visualise last 60 days WIG20 stock price with 20 predictive days
fig, axes = plt.subplots(1, 1)

plt.plot(wig20_total_date.iloc[7233:7314], wig20_total_price[7233:7314], color='red')
plt.axvline(x='2022-05-24', color='b', linestyle='-')

# Reduce the number of plot ticks
for i, tick in enumerate(axes.xaxis.get_ticklabels()):
    if i % 14 != 0:
        tick.set_visible(False)

plt.title('WIG20 Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
