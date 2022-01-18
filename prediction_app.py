
from fbprophet import Prophet
from fbprophet.plot import plot_plotly 
from plotly import graph_objs as go
from datetime import date
import streamlit as st
import yfinance as yf
import pandas as pd


starting_date = '2016-01-01'
date_today= date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

#creating a tuple for stock selection
stock_selection =('TSLA','GME','AMC','AAPL','AMZN','MSFT','BAC','CCL','NVDA','ARKK')

selected_stock=st.selectbox("Select a Stock For Prediction!", stock_selection)
years_prediction = st.slider("Years of prediction you would like to make:", 1,5)

period = years_prediction*365


#@st.cache
def load_data (ticker):
    data=yf.download(ticker,starting_date, date_today)
    data.reset_index (inplace =True)
    return data


data_retrieval = st.text('Retrieve data')

data= load_data(selected_stock)
data_retrieval.text("Stock Data Successfully Retrieved!")

st.subheader('Historical Stock Data')

#putting the data into a dataframe
df=pd.DataFrame(data)

df['MA20'] = df.Close.rolling(20).mean()

st.write(df)


st.write(""" ### Candlestick Stock Chart """)
def plot_chart():
    fig = go.Figure(data=[go.Candlestick(x=df.Date,
                                     open=df.Open, 
                                     high=df.High,
                                     low=df.Low,
                                     close=df.Close)])

    
    fig.add_trace(go.Scatter(x=df.Date, y=df.MA20, name='20-day Moving Average',line=dict(color='blue', width=1)))
    fig.layout.update(autosize=False, width=950,height=593,
                      xaxis_rangeslider_visible=True)
    
    
    st.plotly_chart(fig)


    
plot_chart()    



#Creating a training data set for the prediction model
df_train = df[['Date','Close']]

#renaming the columns to be used in the FB Prophet Prediction Model
df_train.columns = ["ds", "y"]                
model = Prophet ()


model.fit(df_train)
future = model.make_future_dataframe(periods=period)

forecast= model.predict(future)


st.write(""" ### Stock Forecast """)
fig_prediction=plot_plotly(model,forecast)
st.plotly_chart(fig_prediction)



     
     

    
    
    


    
