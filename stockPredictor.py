# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:45:40 2020

"""

import tkinter
import tkinter.messagebox
import datetime
from matplotlib.pyplot import *
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading

top = tkinter.Tk()
top.wm_title("Stock Predictor")

head= tkinter.Label(top, text="Stock Predictor", fg="red", font=("Verdana 10 bold",25))
head.pack()
head.place(x= 180, y= 10)

today = datetime.date.today()
dates = []
prev_dates = []

for i in range(30):
    date = today + datetime.timedelta(days=i+1)
    dates.append(date.strftime("%m-%d"))

for i in range(30):
    date = today + datetime.timedelta(days=0-i)
    prev_dates.append(date.strftime("%m-%d"))

def getStockData():
    tickerSymbol = acr[s.get()-1]
    #get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)
    d = today + datetime.timedelta(days=1)
    #get the historical prices for this ticker
    complete_data = tickerData.history(period='1d', start='1990-1-1', end=d.strftime("%Y-%m-%d"))
    
    complete_data = complete_data[['Close']]
    # Variable for number of days you want to predict ahead of time
    future_pred = 30
    complete_data['Prediction'] = complete_data[['Close']].shift(-1)
    z = complete_data[['Close']][-future_pred:]
    z = np.array(z)
    z = np.hstack(z)
    #print the new data set
    # Create the independent variable or dataset (x)
    # Convert the dataset into a numpy array
    x = np.array(complete_data.drop(['Prediction'],1))
    x = x[:-future_pred]
    # Create the dependent variable or dataset(y)
    # Convert the dataset into a numpy array
    y = np.array(complete_data['Prediction'])
    # Get all the y values except the last 'future_pred' rows (which is 30 currently)
    y = y[:-future_pred]
    # Splitting the dataset into 80% Training set and 20% Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    # training the SVR model on the whole dataset 
    #from sklearn.svm import SVR
    #regressor = SVR(kernel = 'rbf', C=1e3, gamma=0.1)
    #regressor.fit(x_train, y_train)
    # Testing the model using score which returns R^2 of the prediction
    #regressor_confidence = regressor.score(x_test, y_test)
    #print("Regressor Confidence: ", regressor_confidence)
    # Train the Linear Regression model on the Training set
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    # Testing the model using score which returns R^2 of the prediction
    lin_reg_confidence = lin_reg.score(x_test, y_test)
    # Set x_future equal to the last 30 rows of the original data set from Close column
    x_future = np.array(complete_data.drop(['Prediction'],1))[-future_pred:]
    # Linear regression model predictions for the next '30' or future_pred days
    lin_reg_prediction = lin_reg.predict(x_future)
    
    openNewWindow(z, lin_reg_prediction)
    # Support vector regressor model predictions for the next '30' or future_pred days
    #svm_prediction = regressor.predict(x_future)
    #print(svm_prediction)

def callstockdata():
    thr1 = threading.Thread(target = getStockData())
    thr1.start()

def openNewWindow(zz, pp):
    newWindow = tkinter.Toplevel()
    newWindow.geometry("1000x500")
    newWindow.title(stocks[s.get()-1][0] + " Stock Data Prediction")
    newWindow.resizable(width=False, height=False)
    
    data2 = {'Stock Prices History': zz, 'Dates': prev_dates}
    df2 = pd.DataFrame(data2, columns = ['Stock Prices History', 'Dates'])
    figure2 = plt.Figure(figsize=(5,4), dpi=100)
    ax2 = figure2.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure2, newWindow)
    line2.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH)
    df2 = df2[['Stock Prices History','Dates']].groupby('Dates').sum()
    df2.plot(kind='line', legend=True, ax=ax2, color='b',marker='o', fontsize=10)
    ax2.set_title('Stock Prices for last 30 days')
    ax2.set_ylabel("Prices")
    
    data = {'Stock Prices Prediction': pp, 'Dates': dates}
    df1 = pd.DataFrame(data, columns = ['Stock Prices Prediction', 'Dates'])
    figure1 = plt.Figure(figsize=(5,4), dpi=100)
    ax1 = figure1.add_subplot(111)
    line1 = FigureCanvasTkAgg(figure1, newWindow)
    line1.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH)
    df1 = df1[['Stock Prices Prediction','Dates']].groupby('Dates').sum()
    df1.plot(kind='line', legend=True, ax=ax1, color='r',marker='o', fontsize=10)
    ax1.set_title('Stock Prices Predicion')
    ax1.set_ylabel("Prices")
    

s1= tkinter.Label(top, text="------------------------------------------------------------------------------------------------------------------------")
s1.pack()
s1.place(x= 0, y= 50)

s2= tkinter.Label(top, text="------------------------------------------------------------------------------------------------------------------------")
s2.pack()
s2.place(x= 0, y= 65)

s = tkinter.IntVar()

stocks = [
    ("Amazon",1),
    ("Facebook",2),
    ("Microsoft",3),
    ("Google",4),
    ("Apple",5)
]

acr = [
    ("AMZN"),
    ("FB"),
    ("MSFT"),
    ("GOOG"),
    ("AAPL")
]

l1 = tkinter.Label(top, text= """Choose the stock you want to predict the data for""")
l1.config(font=("Times New Roman", 15))
l1.pack()
l1.place(x= 100, y= 120)

r1 = tkinter.Radiobutton(top, 
              text=stocks[0][0], 
              variable=s, 
              value=stocks[0][1])
r1.config(font=("Times New Roman", 12))
r1.pack(anchor=tkinter.W)
r1.place(x = 230, y = 170)

r2 = tkinter.Radiobutton(top, 
              text=stocks[1][0], 
              variable=s, 
              value=stocks[1][1])
r2.config(font=("Times New Roman", 12))
r2.pack(anchor=tkinter.W)
r2.place(x = 230, y = 200)

r3 = tkinter.Radiobutton(top, 
              text=stocks[2][0], 
              variable=s, 
              value=stocks[2][1])
r3.config(font=("Times New Roman", 12))
r3.pack(anchor=tkinter.W)
r3.place(x = 230, y = 230)

r4 = tkinter.Radiobutton(top, 
              text=stocks[3][0], 
              variable=s, 
              value=stocks[3][1])
r4.config(font=("Times New Roman", 12))
r4.pack(anchor=tkinter.W)
r4.place(x = 230, y = 260)

r5 = tkinter.Radiobutton(top, 
              text=stocks[4][0], 
              variable=s, 
              value=stocks[4][1])
r5.config(font=("Times New Roman", 12))
r5.pack(anchor=tkinter.W)
r5.place(x = 230, y = 290)

getData= tkinter.Button(top, text="Get Predictions", command= getStockData)
getData.pack()
getData.place(x= 230, y= 330)

top.geometry('{}x{}'.format(600, 400))
top.resizable(width=False, height=False)
top.mainloop()
