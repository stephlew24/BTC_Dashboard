import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
import keras
import tensorflow as tf
import altair as alt


#Initialize database
client = MongoClient()
db = client.crypto

#Insert Tables
prices = db.prices
block_size = db.blocksize
avg_block_size = db.avgblocksize
n_trans = db.ntrans
n_payments = db.npayments
total_n_trans = db.totalntrans
med_confirm = db.medconfirm
avg_confirm = db.avgconfirm
news = db.news
stats = db.stats

dataset = list(prices.find({}))
price_df = pd.DataFrame(columns=['date','low','high','open','close','volume'])
for i in dataset:
    row = {'date':i['date'],'low':i['low'],'high':i['high'],'open':i['open'],'close':i['close'],'volume':i['vol_fiat']}
    price_df = price_df.append(row, ignore_index=True)
price_df.drop_duplicates(subset=['date'], keep='last', inplace=True)    
price_df = price_df.set_index('date', drop=True).sort_index()

def difference(df, interval=1):
    diff = list()
    for i in range(interval, len(df)):
        value = df[i] - df[i - interval]
        diff.append(value)
    return diff
def returns(df, interval=1):
    returns = list()
    for i in range(interval, len(df)):
        value = np.log(df[i]/df[i - interval])
        returns.append(value)
    return returns
def calc_data(df):    
    df['close_shift'] = pd.Series(difference(df.close, interval=1), index=df.index[1:])
    df['returns'] = pd.Series(returns(df.close, interval=1), index=df.index[1:])
    return df[1:]
def scale_data(df):
    result = calc_data(df)
    scaler = StandardScaler()
    singlediff_columns = ['close_shift']
    X_train_scaled = scaler.fit_transform(result[singlediff_columns][1:])
    return X_train_scaled, scaler
def train_test(df, split=.8):
    X_train_scaled, scaler = scale_data(df)
    train_size = int(len(X_train_scaled) * split)
    test_size = len(X_train_scaled) - train_size
    train, test = X_train_scaled[0:train_size,:], X_train_scaled[train_size:len(X_train_scaled),:]
    return train, test, scaler, X_train_scaled
def process(df, samples=7):
    train, test, scaler, X_train_scaled = train_test(df)
    dataX1 = []
    dataY1 = []
    dataX2 = []
    dataY2 = []
    for i in range(len(train)-samples):
        dataX1.append(train[i:(i+samples)])
        dataY1.append(train[i+samples])
    for i in range(len(test)-samples):
        dataX2.append(test[i:(i+samples)])
        dataY2.append(test[i+samples])
    return np.array(dataX1), (np.array(dataY1).reshape(np.array(dataY1).shape[0],-1)), np.array(dataX2), (np.array(dataY2).reshape(np.array(dataY2).shape[0],-1)), scaler, samples, X_train_scaled
def generate_model(df):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    X_train, y_train, X_test, y_test, scaler, samples, X_train_scaled = process(df)

    METRICS = [
      keras.metrics.MeanSquaredError(name="MSE"),
      keras.metrics.RootMeanSquaredError(name="RMSE"),
      keras.metrics.MeanAbsoluteError(name="MAE"),
      keras.metrics.MeanAbsolutePercentageError(name="MAPE"),
      keras.metrics.MeanSquaredLogarithmicError(name="MSLE"),
    ]
    model = keras.Sequential([
    keras.layers.LSTM(10, activation='relu',input_shape=(X_train.shape[1],X_train.shape[2])),
    keras.layers.Dense(1),])
    model.compile(optimizer='adam',
                  loss='MSE', 
                  metrics=METRICS)

    #Run the model
    epochs = 50
    batch_size = 25

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='MAE', 
        verbose=0,
        patience=10,
        mode='min',
        restore_best_weights=True)

    #base_model = generate_model(X_train)
    base_history = model.fit(X_train,
                             y_train, 
                             epochs=epochs, 
                             batch_size=batch_size,
                             callbacks=[early_stopping],
                             verbose=0)
    result = model, X_train, y_train, X_test, y_test, scaler, samples, X_train_scaled
    return result
def fut_predictions(df, num_prediction=7):
    model, X_train, y_train, X_test, y_test, scaler, samples, X_train_scaled = generate_model(df)
    batch_size = 25
    prediction_list = X_train_scaled[-samples:]
    for _ in range(num_prediction):
        x = prediction_list[-samples:]
        x = x.reshape((1, samples, 1))
        out = model.predict(x, batch_size=batch_size)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[samples-1:]
    prediction_list = scaler.inverse_transform(prediction_list)
    #Set up the index of dates
    last_date = df.index[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_list, prediction_dates
    #Reverse the differencing for predictions
def rev_dif_pred(df, column, interval=1):
    pred, dates = fut_predictions(df)
    pred[0] = df[column][-interval]
    for i in range(1, len(pred)):
        pred[i] = pred[i-interval] - pred[i]
    return pred, dates
@st.cache
def convert_fut_pred(df, column):
    pred, dates = rev_dif_pred(df, column)
    forecast_df = pd.DataFrame(index=dates)
    forecast_df['pred'] = pred
    forecast_df['pred'] = forecast_df['pred'].round(decimals=2)
    return forecast_df
def predictions(df):
    model, X_train, y_train, X_test, y_test, scaler, samples, X_train_scaled = generate_model(df)
    batch_size = 25

    #Reverse scaling
    prediction_train = scaler.inverse_transform(model.predict(X_train, batch_size=batch_size))
    prediction_test = scaler.inverse_transform(model.predict(X_test, batch_size=batch_size))

    #Reshape 
    prediction_train = prediction_train.reshape((-1))
    prediction_test = prediction_test.reshape((-1))
    pred = np.concatenate((prediction_train, prediction_test), axis=0)
    return pred
    #Reverse the differencing for the train set
def rev_dif(df, column, interval=1):
    pred = predictions(df)
    inv_diff = list()
    for i in range(len(df)-len(pred), len(df)):
        value = df[column][i] - pred[(i - (len(df)-len(pred)) + interval) - interval]
        inv_diff.append(value)
    return inv_diff
@st.cache
def convert_pred(df, column):
    pred = rev_dif(df, column)
    result_df = pd.DataFrame(index=df.index[-len(pred):])
    result_df['test'] = df[column][-len(pred):]
    result_df['pred'] = pred
    result_df['pred'] = result_df['pred'].round(decimals=2) 
    return result_df


st.set_page_config(page_icon='💰',
                    layout="wide")

st.sidebar.image("/mnt/c/Users/Steph/Desktop/Bootcamp/Metis/projects/Metis_Projects/Project7/BTC_Logo.png", use_column_width=True)
st.markdown("<h1 style='text-align: center; color: black;'>Bitcoin Dashboard</h1>", unsafe_allow_html=True)
option = st.sidebar.selectbox("", ('Stats', 'Charts', 'Sentiment'))
st.markdown("<hr/>",unsafe_allow_html=True)

if option == "Stats":
    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.markdown("<h2 style='text-align: center; color: black;'>Price USD</h2>", unsafe_allow_html=True)
        if int(list(prices.find({}).sort('date', -1))[2]['close']) > int(list(prices.find({}).sort('date', -1))[1]['close']):
            number1 = (f"⬇️💲{list(prices.find({}).sort('date', -1))[1]['close']:,}")
        elif int(list(prices.find({}).sort('date', -1))[2]['close']) == int(list(prices.find({}).sort('date', -1))[1]['close']):
            number1 = (f"💲{list(prices.find({}).sort('date', -1))[1]['close']:,}")
        else:
            number1 = (f"⬆️💲{list(prices.find({}).sort('date', -1))[1]['close']:,}")
        st.markdown(f"<h1 style='text-align: center; color: gold;'>{number1}</h1>", unsafe_allow_html=True)

    with metric2:
        st.markdown("<h2 style='text-align: center; color: black;'>Trade volume (USD)</h2>", unsafe_allow_html=True)
        if int(list(stats.find({}))[-2]['trade_volume_usd']) > int(list(stats.find({}))[-1]['trade_volume_usd']):
            number2 = (f"⬇️{int(list(stats.find({}))[-1]['trade_volume_usd']):,}")
        elif int(list(stats.find({}))[-2]['trade_volume_usd']) == int(list(stats.find({}))[-1]['trade_volume_usd']):
            number2 = (f"{int(list(stats.find({}))[-1]['trade_volume_usd']):,}")
        else:
            number2 = (f"⬆️{int(list(stats.find({}))[-1]['trade_volume_usd']):,}") 
        st.markdown(f"<h1 style='text-align: center; color: gold;'>{number2}</h1>", unsafe_allow_html=True)

    with metric3:
        st.markdown("<h2 style='text-align: center; color: black;'>Trade Fees (BTC)</h2>", unsafe_allow_html=True)
        if int(list(stats.find({}))[-2]['total_fees_btc']) > int(list(stats.find({}))[-1]['total_fees_btc']):
            number3 = (f"⬇️{int(list(stats.find({}))[-1]['total_fees_btc']):,}") 
        elif int(list(stats.find({}))[-2]['total_fees_btc']) == int(list(stats.find({}))[-1]['total_fees_btc']):
            number3 = (f"{int(list(stats.find({}))[-1]['total_fees_btc']):,}")
        else:
            number3 = (f"⬆️{int(list(stats.find({}))[-1]['total_fees_btc']):,}")
        st.markdown(f"<h1 style='text-align: center; color: gold;'>{number3}</h1>", unsafe_allow_html=True)

    st.markdown("<hr/>",unsafe_allow_html=True)


    metric01, metric02, metric03 = st.columns(3)

    with metric01:
        st.markdown("<h2 style='text-align: center; color: black;'>Hash Rate</h2>", unsafe_allow_html=True)
        if int(list(stats.find({}))[-2]['hash_rate']) > int(list(stats.find({}))[-1]['hash_rate']):
            number01 = (f"⬇️{int(list(stats.find({}))[-1]['hash_rate']):,}")
        elif int(list(stats.find({}))[-2]['hash_rate']) == int(list(stats.find({}))[-1]['hash_rate']):
            number01 = (f"{int(list(stats.find({}))[-1]['hash_rate']):,}")
        else: 
            number01 = (f"⬆️{int(list(stats.find({}))[-1]['hash_rate']):,}")
        st.markdown(f"<h1 style='text-align: center; color: silver;'>{number01}</h1>", unsafe_allow_html=True)
        
    with metric02:
        st.markdown("<h2 style='text-align: center; color: black;'>Average Block Size</h2>", unsafe_allow_html=True)
        if round(list(avg_block_size.find({}))[-2]['data'],2) > round(list(avg_block_size.find({}))[-1]['data'],2):
            number02 = (f"⬇️{round(list(avg_block_size.find({}))[-1]['data'],2)}")
        elif round(list(avg_block_size.find({}))[-2]['data'],2) == round(list(avg_block_size.find({}))[-1]['data'],2):
            number02 = (f"{round(list(avg_block_size.find({}))[-1]['data'],2)}")
        else:
            number02 = (f"⬆️{round(list(avg_block_size.find({}))[-1]['data'],2)}")
        st.markdown(f"<h1 style='text-align: center; color: silver;'>{number02}</h1>", unsafe_allow_html=True)

    with metric03:
        st.markdown("<h2 style='text-align: center; color: black;'>Total Transactions</h2>", unsafe_allow_html=True)
        if int(list(total_n_trans.find({}))[-2]['data']) > int(list(total_n_trans.find({}))[-1]['data']):
            number03 = (f"⬇️{int(list(total_n_trans.find({}))[-1]['data']):,}")
        elif int(list(total_n_trans.find({}))[-2]['data']) == int(list(total_n_trans.find({}))[-1]['data']):
            number03 = (f"{int(list(total_n_trans.find({}))[-1]['data']):,}")
        else:
            number03 = (f"⬆️{int(list(total_n_trans.find({}))[-1]['data']):,}")
        st.markdown(f"<h1 style='text-align: center; color: silver;'>{number03}</h1>", unsafe_allow_html=True)

elif option == "Charts":
    prediction = convert_fut_pred(price_df, 'close')
    pricelog = alt.Chart(price_df.reset_index(),title="Bitcoin Price Chart Log Scale").mark_line().encode(
        alt.X('date:T',
              axis=alt.Axis(
                  format='%Y-%m',
                  labelAngle=-45),
             ),
        alt.Y('close:Q',
              scale= alt.Scale(type= 'log', base=2)),
         color=alt.value('gold'),
         tooltip=[alt.Tooltip('date:T'),
                  alt.Tooltip('low:Q', format='0,.2f'),
                  alt.Tooltip('high:Q', format='0,.2f'),
                  alt.Tooltip('open:Q', format='0,.2f'),
                  alt.Tooltip('close:Q', format='0,.2f'),
                  alt.Tooltip('volume:Q', format='0,.0f')]
    ).properties(
        width=600, height=400).configure_axis(
     labelFontSize=16,
     titleFontSize=16).configure_title(fontSize=24)

    volumelog = alt.Chart(price_df.reset_index(),title="Bitcoin volume Chart").mark_line().encode(
        alt.X('date:T',
              axis=alt.Axis(
                  format='%Y',
                  labelAngle=-45)
             ),
        y='volume:Q',
        color=alt.value('silver'),
        tooltip=[alt.Tooltip('date:T'),
                 alt.Tooltip('volume:Q', format='0,.2f')]
        ).properties(
        width=600, height=200).configure_axis(
     labelFontSize=16,
     titleFontSize=16).configure_title(fontSize=24)
    
    price = alt.Chart(price_df.reset_index(),title="Bitcoin Price Chart (Log Scale)").mark_line().encode(
    alt.X('date:T',
          axis=alt.Axis(
              format='%Y-%m',
              labelAngle=-45),
         ),
     y='close:Q',
     color=alt.value('gold'),
     tooltip=[alt.Tooltip('date:T'),
              alt.Tooltip('low:Q', format='0,.2f'),
              alt.Tooltip('high:Q', format='0,.2f'),
              alt.Tooltip('open:Q', format='0,.2f'),
              alt.Tooltip('close:Q', format='0,.2f'),
              alt.Tooltip('volume:Q', format='0,.0f')]
    ).properties(
        width=600, height=400).configure_axis(
     labelFontSize=16,
     titleFontSize=16).configure_title(fontSize=24)

    volume = alt.Chart(price_df.reset_index(),title="Bitcoin volume Chart").mark_line().encode(
        alt.X('date:T',
              axis=alt.Axis(
                  format='%Y',
                  labelAngle=-45)
             ),
        y='volume:Q',
        color=alt.value('silver'),
        tooltip=[alt.Tooltip('date:T'),
                 alt.Tooltip('volume:Q', format='0,.2f')]
        ).properties(
        width=600, height=200).configure_axis(
     labelFontSize=16,
     titleFontSize=16).configure_title(fontSize=24)

    st.altair_chart(pricelog, use_container_width=True)
    st.altair_chart(volumelog, use_container_width=True)

    
    st.markdown("<hr/>",unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: black;'><b>Tommorow's Price Prediction (USD)</b></h2>", unsafe_allow_html=True)
    number04 = (f"💲{prediction.pred[1]:,}")
    st.markdown(f"<h1 style='text-align: center; color: gold;'>{number04}</h1>", unsafe_allow_html=True)


    

