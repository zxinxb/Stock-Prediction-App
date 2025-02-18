import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
import sys
from datetime import date


START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('SOLFINTECH')

my_ticker = ('RR.L', 'AZN.L', 'VOD.L', 'BHP.L', 'HSBA.L', 'IHG.L', 'KGF.L','SHEL.L', 'TSCO.L', 'STAN.L',
          'SGRO.L', 'ULVR.L', 'JD.L', 'MNDI.L', 'PSON.L', 'PHNX.L', 'RIO.L', 'REL.L', 'ADM.L', 'ABF.L', 'AV.L')
select_stock = st.selectbox('Select Stock Ticker', my_ticker)

# Download data for the selected period
stock_data = yf.download(select_stock, START, TODAY)

st.subheader(f'Data for {select_stock} from {START} to {TODAY}')
st.write(stock_data.tail())


# Input for the number of days to predict
num_days = st.number_input('Enter Number of Days to Predict', value=1, min_value=1, step=1)

try:
    end_date_predicted = (pd.to_datetime(TODAY) + pd.DateOffset(days=num_days)).strftime('%Y-%m-%d')

    # Ensure we don't exceed the available data
    selected_num_days = min(num_days, len(stock_data))
    start_date = stock_data.index[-selected_num_days].strftime("%Y-%m-%d")

    st.subheader(f'Closing Price Chart for {select_stock}')
    st.line_chart(stock_data['Close'])

except Exception as e:
    st.error(f"An error occurred:{e}")


# Visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(stock_data.Close)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100ma & 200ma')
ma_100 = stock_data.Close.rolling(100).mean()
ma_200 = stock_data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma_100, 'r')
plt.plot(ma_200, 'g')
plt.plot(stock_data.Close, 'b')
st.pyplot(fig)


# Splitting data into training and testing

data_training = pd.DataFrame(stock_data.Close[0:int(len(stock_data)*0.70)])
data_testing = pd.DataFrame(stock_data.Close[int(len(stock_data)*0.70): int(len(stock_data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_scale = scaler.fit_transform(data_training)

# Load my model
model = load_model('Stock Prediction Model.keras')

#testing 
past_100_days = data_training.tail(100)
data_testing = pd.concat([past_100_days, data_testing], ignore_index=True)
data_test_scale  =  scaler.fit_transform(data_testing)

#Lets create 2 arrays for array slicing
X_test = []
y_test = []

for i in range(100, data_test_scale.shape[0]):
    X_test.append(data_test_scale[i-100: i])
    y_test.append(data_test_scale[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

y_predict = model.predict(X_test)

scale =1/scaler.scale_
y_predict = y_predict*scale
y_test = y_test*scale


# Input for investment amount
investment_amount = st.number_input('Enter Investment Amount', value=0.0)

# Calculate final predicted price and profit/loss
final_predicted_price = y_predict[-1, 0]  # Extract scalar value from ndarray
profit_loss = (final_predicted_price - investment_amount)

# Display number of shares
num_shares = investment_amount / final_predicted_price
st.info(f'Number of Shares: {num_shares:.2f}')

# Simulate trading
capital = investment_amount
shares = 0

for i, row in stock_data.iterrows():
    if row['Close'] > row['Open']:
        shares_to_buy = capital // row['Close']
        shares += shares_to_buy
        capital -= shares_to_buy * row['Close']
    else:
        capital += shares * row['Close']
        shares = 0


# Calculate final profit/loss
final_balance = capital + shares * stock_data.iloc[-1]['Close']
profit_loss = final_balance - investment_amount

# Display results
st.subheader('Stock Results')
st.write(f'Initial Investment: ${investment_amount:.2f}')
st.write(f'Final Balance: ${final_balance:.2f}')
st.write(f'Profit/Loss: ${profit_loss:.2f}')

# Plot Buy and Sell signals
st.subheader('Buy and Sell Signals')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(stock_data.index, stock_data['Close'], label='Close Price')
ax.plot(stock_data[stock_data['Close'] > stock_data['Open']].index, stock_data[stock_data['Close'] > stock_data['Open']]['Close'], '^', markersize=10, color='g', label='Buy Signal')
ax.plot(stock_data[stock_data['Close'] <= stock_data['Open']].index, stock_data[stock_data['Close'] <= stock_data['Open']]['Close'], 'v', markersize=10, color='r', label='Sell Signal')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)


# Final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(10, 8))
plt.plot(y_predict, 'r', label = 'Predicted Price')
plt.plot(y_test, 'g', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# User selects stock tickers for comparison
selected_comparison_tickers = st.multiselect('Select Stock Tickers for Comparison', ["RR.L", "AZN.L", "VOD.L", "BHP.L", "HSBA.L", 
                                                                                     "IHG.L", "KGF.L", "SHEL.L", "TSCO.L", "STAN.L",
                                                                                     "SGRO.L", "ULVR.L", "JD.L", "MNDI.L", "PSON.L", 
                                                                                     "PHNX.L", "RIO.L", "REL.L", "ADM.L", "ABF.L", "AV.L"])

# Define the thresholds for buy and sell signals (5%)
buy_threshold = 0.05
sell_threshold = 0.05

# Display the comparison result
st.subheader('Comparison Result')

for ticker in selected_comparison_tickers:
    try:
        comparison_df = yf.download(ticker, START, TODAY)  # Download data for the selected ticker

        # Calculate profit or loss based on your strategy
        capital_comp = investment_amount
        shares_comp = 0

        signals_comp = []

        for i, row in comparison_df.iterrows():
            if row['Close'] >= (1 + buy_threshold) * final_predicted_price:
                signals_comp.append('Buy')
            elif row['Close'] <= (1 - sell_threshold) * final_predicted_price:
                signals_comp.append('Sell')
            else:
                signals_comp.append('Hold')

        comparison_df['Signal'] = signals_comp

        for i, row in comparison_df.iterrows():
            if row['Signal'] == 'Buy':
                shares_to_buy_comp = capital_comp // row['Close']
                shares_comp += shares_to_buy_comp
                capital_comp -= shares_to_buy_comp * row['Close']
            elif row['Signal'] == 'Sell':
                capital_comp += shares_comp * row['Close']
                shares_comp = 0

        final_balance_comp = capital_comp + shares_comp * comparison_df.iloc[-1]['Close']
        comparison_profit_loss = final_balance_comp - investment_amount

        st.write(f'Profit/Loss for {ticker}: ${comparison_profit_loss:.2f}')
    except Exception as e:
        st.error(f"An error occurred for {ticker}: {e}")
