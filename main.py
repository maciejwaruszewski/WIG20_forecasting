# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Importing the training set
dataset_train = pd.read_csv('wig20_d.csv')
new_cols = ['Data', 'Zamkniecie', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Wolumen']
dataset_train = dataset_train.reindex(columns=new_cols)
# '.values' need the 2nd Column Opening Price as a Numpy array (not vector)
# '1:2' is used because the upper bound is ignored
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
# Use Normalization (versus Standardization) for RNNs with Sigmoid Activation Functions
# 'MinMaxScalar' is a Normalization Library
# 'feature_range = (0,1)' makes sure that training data is scaled to have values between 0 and 1
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps (look back 60 days) and 1 output
# This tells the RNN what to remember (Number of timesteps) when predicting the next Stock Price
# The wrong number of timesteps can lead to Overfitting or bogus results
# 'x_train' Input with 60 previous days' stock prices
X_train = []
# 'y_train' Output with next day's stock price
y_train = []
for i in range(60, 7292):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping (add more dimensions)
# This lets you add more indicators that may potentially have corelation with Stock Prices
# Keras RNNs expects an input shape (Batch Size, Timesteps, input_dim)
# '.shape[0]' is the number of Rows (Batch Size)
# '.shape[1]' is the number of Columns (timesteps)
# 'input_dim' is the number of factors that may affect stock prices
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Show the dataset we're working with

# Part 2 - Building the RNN
# Building a robust stacked LSTM with dropout regularization

# Importing the Keras libraries and packages
# Initialising the RNN
# Regression is when you predict a continuous value
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# 'units' is the number of LSTM Memory Cells (Neurons) for higher dimensionality
# 'return_sequences = True' because we will add more stacked LSTM Layers
# 'input_shape' of x_train
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# 20% of Neurons will be ignored (10 out of 50 Neurons) to prevent Overfitting
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
# Not need to specify input_shape for second Layer, it knows that we have 50 Neurons from the previous layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
# This is the last LSTM Layer. 'return_sequences = false' by default so we leave it out.
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
# 'units = 1' because Output layer has one dimension
regressor.add(Dense(units=1))

# Compiling the RNN
# Keras documentation recommends 'RMSprop' as a good optimizer for RNNs
# Trial and error suggests that 'adam' optimizer is a good choice
# loss = 'mean_squared_error' which is good for Regression vs. 'Binary Cross Entropy' previously used for Classification
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
# 'X_train' Independent variables
# 'y_train' Output Truths that we compare X_train to.
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('wig20_d_real.csv')
new_cols = ['Data', 'Zamkniecie', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Wolumen']
dataset_test = dataset_test.reindex(columns=new_cols)
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# We need 60 previous inputs for each day of the Test_set in 2017
# Combine 'dataset_train' and 'dataset_test'
# 'axis = 0' for Vertical Concatenation to add rows to the bottom
dataset_total = pd.concat((dataset_train['Zamkniecie'], dataset_test['Zamkniecie']), axis=0)
# Extract Stock Prices for Test time period, plus 60 days previous
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# 'reshape' function to get it into a NumPy format
inputs = inputs.reshape(-1, 1)
# Inputs need to be scaled to match the model trained on Scaled Feature
inputs = sc.transform(inputs)
# The following is pasted from above and modified for Testing, romove all 'Ys'
X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
# We need a 3D input so add another dimension
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Predict the Stock Price
predicted_stock_price = regressor.predict(X_test)
# We need to inverse the scaling of our prediction to get a Dollar amount
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


predicted_otwarcie = pd.DataFrame(predicted_stock_price, columns={'Otwarcie'})
predicted_najwyzszy = pd.DataFrame(predicted_stock_price, columns={'Najwyzszy'})
predicted_najnizszy = pd.DataFrame(predicted_stock_price, columns={'Najnizszy'})
predicted_zamkniecie = pd.DataFrame(predicted_stock_price, columns={'Zamkniecie'})
predicted_wolumen = pd.DataFrame(predicted_stock_price, columns={'Wolumen'})


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

wig20_forecast.to_csv('wig20_forecast.csv')

wig20_total_date = pd.concat((dataset_train['Data'], wig20_forecast['Data']), axis=0)
wig20_total_price = pd.concat((dataset_train['Zamkniecie'], wig20_forecast['Zamkniecie']), axis=0)
wig20_total_open = pd.concat((dataset_train['Otwarcie'], wig20_forecast['Otwarcie']), axis=0)
wig20_total_max = pd.concat((dataset_train['Najwyzszy'], wig20_forecast['Najwyzszy']), axis=0)
wig20_total_min = pd.concat((dataset_train['Najnizszy'], wig20_forecast['Najnizszy'], ), axis=0)
wig20_total_volumen = pd.concat((dataset_train['Wolumen'], wig20_forecast['Wolumen']), axis=0)

wig20_total_date.to_csv('wig20_forecast.csv')

plt.plot(wig20_total_date, wig20_total_price, color='red')
plt.axvline(x='2022-05-24', color='b', linestyle='-')
plt.title('WIG20 Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
