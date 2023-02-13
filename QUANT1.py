#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas_ta as pta
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from datetime import timedelta
from tensorflow.keras.callbacks import TensorBoard
import time
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import normalize


# In[12]:


class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = pd.DataFrame(data=None, columns=['Date','RSI','WILLR','WMA','EMA','SMA','HMA','TEMA','CCI','CMO','MACD','PPO','ROC','CMF','ADX','PSAR'])
        self.yfticker = yf.Ticker(self.ticker)
        self.tickerdf = self.yfticker.history(period='max')
        self.tickerdf = self.tickerdf.tz_localize(None)
        self.counter = 0
        self.total = 0
        
    def printdf(self):
        print(self.tickerdf)
        
    def UpdateTicker(self):
        self.tickerHistory = yf.Ticker(self.ticker)
        self.tickerdf = self.yfticker.history(period='max')
        self.tickerdf = self.tickerdf.tz_localize(None)
    
    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))*255
    
    def Convert2ImageLarge(self, twoDArray):
        imArrNormalized = np.apply_along_axis(self.NormalizeData, 0, twoDArray)
        imArrNormalizedPixels = np.array(imArrNormalized, dtype=np.uint8)
        #img = Image.fromarray(imArrNormalizedPixels,'L')
        #img = img.rotate(90, Image.NEAREST, expand = 1)
        return imArrNormalizedPixels.loc[:, self.df1.columns != 'Date']
            
    def RSI(self,date):
        return pta.rsi(self.tickerdf['Close'], length = 20).loc[date]
        
    #Williams %R Indicator
    def WILLR(self,date):
        return pta.willr(self.tickerdf['High'],self.tickerdf['Low'],self.tickerdf['Close'],20).loc[date]
    
    #Weighted moving average
    def WMA(self,date):
        return pta.wma(self.tickerdf['Close'],20).loc[date]
    
    #Exponential moving average
    def EMA(self, date):
        return pta.ema(self.tickerdf['Close'],20).loc[date]
    
    #Simple moving average
    def SMA(self,date):
        return pta.sma(self.tickerdf['Close'],20).loc[date]
    
    #Hull exponential moving average
    def HMA(self, date):
        return pta.hma(self.tickerdf['Close'],20).loc[date]
    
    #Triple exponential moving average
    def TEMA(self, date):
        return pta.tema(self.tickerdf['Close'],20).loc[date]
    
    #Commodity Channel Index
    def CCI(self, date):
        return pta.cci(self.tickerdf['High'],self.tickerdf['Low'],self.tickerdf['Close'],20,0.015).loc[date]
    
    #Chande Momentum Oscillator
    def CMO(self, date):
        return pta.cmo(self.tickerdf['Close'],20).loc[date]
    
    #Moving Average Convergence/Divergence Oscillator
    def MACD(self, date):
        return pta.macd(self.tickerdf['Close'],12,26,20).loc[date]['MACDh_12_26_20']
    
    #Percentage Price Oscillator
    def PPO(self, date):
        return pta.ppo(self.tickerdf['Close'],26,12,20).loc[date]['PPOh_12_26_20']
    
    #Rate of change
    def ROC(self, date):
        return pta.roc(self.tickerdf['Close'],20).loc[date]
    
    #Chaikin Money Flow
    def CMF(self, date):
        return pta.cmf(self.tickerdf['High'],self.tickerdf['Low'],self.tickerdf['Close'],self.tickerdf['Volume']).loc[date]
    
    #Average Directional Movement Index
    def ADX(self, date):
        return pta.adx(self.tickerdf['High'],self.tickerdf['Low'],self.tickerdf['Close'],20).loc[date]['ADX_20']
    
    #Parabolic Stop and Reverse
    def PSAR(self, date):
        PSAR = pta.psar(self.tickerdf['High'],self.tickerdf['Low'],self.tickerdf['Close'])
        PSAR['Merged'] = PSAR['PSARl_0.02_0.2'].fillna(PSAR['PSARs_0.02_0.2'])
        return PSAR['Merged'].loc[date]
    
    #RSI, WILLR, WMA, EMA, SMA, HMA, TEMA, CCI, CMO, MACD, PPO, ROC, CMF, ADX, PSARM
    def CreateTrainingImage(self, date):
        self.counter = self.counter + 1
        self.df['Date'] = pd.date_range(end=date, periods=20, freq='D')
        self.df['RSI'] = self.df['Date'].apply(self.RSI)
        self.df['WILLR'] = self.df['Date'].apply(self.WILLR)
        self.df['WMA'] = self.df['Date'].apply(self.WMA)
        self.df['EMA'] = self.df['Date'].apply(self.EMA)
        self.df['SMA'] = self.df['Date'].apply(self.SMA)
        self.df['HMA'] = self.df['Date'].apply(self.HMA)
        self.df['TEMA'] = self.df['Date'].apply(self.TEMA)
        self.df['CCI'] = self.df['Date'].apply(self.CCI)
        self.df['CMO'] = self.df['Date'].apply(self.CMO)
        self.df['MACD'] = self.df['Date'].apply(self.MACD)
        self.df['PPO'] = self.df['Date'].apply(self.PPO)
        self.df['ROC'] = self.df['Date'].apply(self.ROC)
        self.df['CMF'] = self.df['Date'].apply(self.CMF)
        self.df['ADX'] = self.df['Date'].apply(self.ADX)
        self.df['PSAR'] = self.df['Date'].apply(self.PSAR)
        
        print('Printing Image:',self.counter,'/',self.total,'date:',date)
        return self.Convert2ImageLarge(self.df.loc[:, self.df.columns != 'Date'])
        #return self.df1.loc[:, self.df1.columns != 'Date']
        #return normalize([self.df.loc[:, self.df.columns != 'Date'].to_numpy()])
    
    def CreateTrainingImagedf(self, startdate, enddate):
        startdate2 = pd.to_datetime(startdate)
        enddate2 = pd.to_datetime(enddate)
        self.total = enddate2 - startdate2
        self.total = self.total + timedelta(days=1)
        imagedf = pd.DataFrame(data=None, columns=['Date','Image'])
        imagedf['Date'] = pd.date_range(startdate, enddate, freq='D')
        imagedf['Image'] = imagedf['Date'].apply(self.CreateTrainingImage)
        
        return imagedf['Image']
    
    def CreateTrainingLabels(self, startdate, enddate):
        startdate = pd.to_datetime(startdate)
        enddate = pd.to_datetime(enddate)
        enddate = enddate+pd.DateOffset(days=1)
        labeldf = self.yfticker.history(start=startdate, end=enddate)
        labeldf['Close_Shifted'] = labeldf['Close'].shift(-1)
        labeldf['Move'] = np.where(labeldf['Close'] < labeldf['Close_Shifted'], True, False)
        
        return labeldf['Move']
    
    def Labelling(self, startdate, enddate):
        startdate = pd.to_datetime(startdate)
        enddate = pd.to_datetime(enddate)
        enddate = enddate + timedelta(days=1)
        windowSize = 11
        labeldf = self.yfticker.history(start=startdate, end=enddate)
        labeldf['Close_Shifted -1'] = labeldf['Close'].shift(-1)
        labeldf['Close_Shifted -2'] = labeldf['Close'].shift(-2)
        labeldf['Close_Shifted -3'] = labeldf['Close'].shift(-3)
        labeldf['Close_Shifted -4'] = labeldf['Close'].shift(-4)
        labeldf['Close_Shifted -5'] = labeldf['Close'].shift(-5)
        labeldf['Close_Shifted 1'] = labeldf['Close'].shift(1)
        labeldf['Close_Shifted 2'] = labeldf['Close'].shift(2)
        labeldf['Close_Shifted 3'] = labeldf['Close'].shift(3)
        labeldf['Close_Shifted 4'] = labeldf['Close'].shift(4)
        labeldf['Close_Shifted 5'] = labeldf['Close'].shift(5)
        subset = ['Close','Close_Shifted -1','Close_Shifted -2','Close_Shifted -3','Close_Shifted -4','Close_Shifted -5','Close_Shifted 1','Close_Shifted 2','Close_Shifted 3','Close_Shifted 4','Close_Shifted 5']
        closes = labeldf[subset]
        labeldf['Max'] = labeldf[subset].max(axis = 1)
        labeldf['Min'] = labeldf[subset].min(axis = 1)
        labeldf['Sell'] = np.where(labeldf['Close'] == labeldf['Max'],1,0)
        labeldf['Buy'] = np.where(labeldf['Close'] == labeldf['Min'],1,0)
        
        def moveCondition(row):
            if row['Sell'] == 1:
                return 'Sell'
            if row['Buy'] == 1:
                return 'Buy'
            else:
                return 'Hold'
        
        labeldf['Move'] = labeldf.apply (lambda row: moveCondition(row), axis=1)
        
        return labeldf['Move']
    
    def financialAnalysis(self, df, startingCapital):
        df = df.tz_localize(None)
        dfmoves = df[df['Labels'] != 'Hold']
        i = 0
        capital = startingCapital           
        status = 'Out'
        shareValue = 0
        numberShares = 0
        net = 0
        for index, row in dfmoves.iterrows():
            if row['Labels'] == 'Buy' and status == 'Out':
                shareValue = self.tickerdf.Close.loc[index]
                numberShares = capital/shareValue
                capital = 0
                status = 'In'
                print('Buying',numberShares,'shares at',shareValue,'on',index)
            elif row['Labels'] == 'Sell' and status == 'In':
                shareValue = self.tickerdf.Close.loc[index]
                capital = numberShares*(shareValue)
                print('Selling',numberShares,'shares at',shareValue,'on',index)
                numberShares = 0
                status = 'Out'
        
        if capital != 0:
            net = capital
        else:
            shareValue = self.tickerdf.Close.iloc[dfmoves.tail(1).index.item()]
            net = numberShares*shareValue
        
        dict = {'Ticker':[self.ticker],
        'Begin Date':[dfmoves.head(1).index.item()],
        'End Date':[dfmoves.tail(1).index.item()],
        'Start Capital':[startingCapital],
        'QUANT Return':[net/startingCapital],
        'Buy&Hold Return':[(self.tickerdf.Close.loc[dfmoves.tail(1).index.item()])/(self.tickerdf.Close.loc[dfmoves.head(1).index.item()])]
       }
        
        returns = pd.DataFrame(dict)
        
        return returns
        


# In[10]:


#Create Images

BTC = Stock('BTC-USD')
BTC.UpdateTicker()
training_df = pd.DataFrame(data=None, columns=['Labels','Images'])
training_df['Labels'] = BTC.Labelling('20211111','20211118')
training_df['Images']=pd.Series(BTC.CreateTrainingImagedf('20211111','20211118'))

#imgList = BTC.CreateTrainingImagedf('20170101','20211118').tolist()
#training_df['Images'] = imgList


# In[13]:


BTC=Stock('BTC-USD')
BTC.UpdateTicker()
img = BTC.CreateTrainingImage('20211111')


# In[9]:


print(img.shape)
img.head()
#img.to_csv(index = False)


# In[ ]:


BTC=Stock('BTC-USD')
BTC.UpdateTicker()
test_df = pd.DataFrame(data=None, columns=['Labels','Images'])
test_df['Labels'] = BTC.Labelling('20211119','20221119')
imgList = BTC.CreateTrainingImagedf('20211119','20221119').tolist()
test_df['Images'] = imgList


# In[ ]:


print(BTC.financialAnalysis(training_df,100000))


# In[ ]:


#Save Images to Drive
buyCounter = 0
sellCounter = 0
holdCounter = 0
edgeCounter = 0

print(training_df.shape[0])
for i in range(training_df.shape[0]):
    name='img'+str(i)+'.png'
    if training_df.Labels[i] == 'Buy':
        training_df.Images[i].save('trainingImages2/Buy/'+name)
        buyCounter += 1
    elif training_df.Labels[i] == 'Sell':
        training_df.Images[i].save('trainingImages2/Sell/'+name)
        sellCounter += 1
    elif training_df.Labels[i] == 'Hold':
        training_df.Images[i].save('trainingImages2/Hold/'+name)
        holdCounter += 1
    else:
        edgeCounter += 1
#print(buyCounter,sellCounter,holdCounter, edgeCounter)
#print(buyCounter + sellCounter + holdCounter + edgeCounter)


# In[ ]:


#Save Images to Drive
buyCounter = 0
sellCounter = 0
holdCounter = 0
edgeCounter = 0

print(test_df.shape[0])
for i in range(test_df.shape[0]):
    name='img'+str(i)+'.png'
    if test_df.Labels[i] == 'Buy':
        test_df.Images[i].save('testImages2/Buy/'+name)
        buyCounter += 1
    elif test_df.Labels[i] == 'Sell':
        test_df.Images[i].save('testImages2/Sell/'+name)
        sellCounter += 1
    elif test_df.Labels[i] == 'Hold':
        test_df.Images[i].save('testImages2/Hold/'+name)
        holdCounter += 1
    else:
        edgeCounter += 1


# In[4]:


image_size = (20, 15)


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "trainingImages2",
    labels='inferred',
    label_mode = 'categorical',
    validation_split=0.1,
    subset = "training",
    image_size=image_size,
    seed = 200
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "trainingImages2",
    labels='inferred',
    label_mode = 'categorical',
    validation_split=0.1,
    subset="validation",
    seed = 200,
    image_size=image_size
    
)


# In[5]:


train_ds = train_ds.prefetch(buffer_size=32)

#%%

# read csv file and create df files
train_df = pd.read_csv("AXP20122018_train.csv", header=None, index_col=None, delimiter=';')
test_df = pd.read_csv("AXP20192023_test.csv", header=None, index_col=None, delimiter=';')

print("Before data imbalance")
l0_train = train_df.loc[train_df[300] == 0]
l1_train = train_df.loc[train_df[300] == 1]
l2_train = train_df.loc[train_df[300] == 2]
l0_train_size = l0_train.shape[0]
l1_train_size = l1_train.shape[0]
l2_train_size = l2_train.shape[0]
print('train_df => l0_size:',l0_train_size,'l1_size:',l1_train_size,'l2_size:',l2_train_size)
l0_test = test_df.loc[test_df[300] == 0]
l1_test = test_df.loc[test_df[300] == 1]
l2_test = test_df.loc[test_df[300] == 2]
l0_test_size = l0_test.shape[0]
l1_test_size = l1_test.shape[0]
l2_test_size = l2_test.shape[0]
print('test_df => l0_size:',l0_test_size,'l1_size:',l1_test_size,'l2_size:',l2_test_size)

# calculate the number of labels and find ratios
l0_l1_ratio = (l0_train_size//l1_train_size)
l0_l2_ratio = (l0_train_size//l2_train_size)
l1_l0_ratio = (l1_train_size//l0_train_size)
l1_l2_ratio = (l1_train_size//l2_train_size)
l2_l0_ratio = (l2_train_size//l0_train_size)
l2_l1_ratio = (l2_train_size//l1_train_size)
print("l0_l1_ratio:",l0_l1_ratio)
print("l0_l2_ratio:",l0_l2_ratio)
print("l1_l0_ratio:",l1_l0_ratio)
print("l1_l2_ratio:",l1_l2_ratio)
print("l2_l0_ratio:",l2_l0_ratio)
print("l2_l1_ratio:",l2_l1_ratio)
#if there is data imbalance, solution of data imbalance in training set
#data imbalance #0/#1>1

# to solve data imbalance
l0_new = pd.DataFrame()
l1_new = pd.DataFrame()
l2_new = pd.DataFrame()
for idx, row in train_df.iterrows():
     if row[300] == 0:
        for i in range(2):
             l0_new = l0_new.append(row)
     if row[300] == 1:
         for i in range(2):
             l1_new = l1_new.append(row)
     if row[300] == 2:
         for i in range(2):
             l2_new = l2_new.append(row)

train_df = train_df.append(l0_new)
train_df = train_df.append(l1_new)
train_df = train_df.append(l2_new)
#
# shuffle
train_df = shuffle(train_df)


print("After data imbalance")
l0_train = train_df.loc[train_df[300] == 0]
l1_train = train_df.loc[train_df[300] == 1]
l2_train = train_df.loc[train_df[300] == 2]
l0_train_size = l0_train.shape[0]
l1_train_size = l1_train.shape[0]
l2_train_size = l2_train.shape[0]
print('train_df => l0_size:',l0_train_size,'l1_size:',l1_train_size,'l2_size:',l2_train_size)
l0_test = test_df.loc[test_df[300] == 0]
l1_test = test_df.loc[test_df[300] == 1]
l2_test = test_df.loc[test_df[300] == 2]
l0_test_size = l0_test.shape[0]
l1_test_size = l1_test.shape[0]
l2_test_size = l2_test.shape[0]
print('test_df => l0_size:',l0_test_size,'l1_size:',l1_test_size,'l2_size:',l2_test_size)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

print("train_df size: ", train_df.shape)

# parameters of the cnn
params = {"input_w": 15, "input_h": 20, "num_classes": 3, "batch_size": 1024, "epochs": 100}

#call cnn
#predictions, test_labels, test_prices = train_cnn(train_df, test_df, params)

#results of the cnn
#result_df = pd.DataFrame({"prediction": np.argmax(predictions, axis=1),
#                           "test_label":np.argmax(test_labels, axis=1),
#                           "test_price":test_prices})
#result_df.to_csv("cnn_result.csv", sep=';', index=None)




# In[6]:


def make_model(input_shape, num_classes):
    
    
    inputs = keras.Input(shape=input_shape)
    #
    model = Sequential(layers.Rescaling(1./255))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(20, 15, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    #model.compile(loss=keras.losses.categorical_crossentropy,
                  #optimizer=keras.optimizers.Adadelta(),
                  #metrics=['accuracy', 'mae', 'mse'])
    
    return model

model = make_model(input_shape=image_size + (1,), num_classes=3)


# In[9]:


NAME = 'QUANT-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model.compile(
    optimizer=keras.optimizers.Adadelta(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
# outputLabels = np.array(['Buy','Hold','Sell'])
# outputs = training_df['Labels'].to_numpy()
# classWeight = compute_class_weight('balanced', outputLabels, outputs) 
# classWeight = dict(enumerate(classWeight))

model.fit(
    train_df, epochs=50, callbacks=[tensorboard], validation_data=test_df
)

save_model(model, "model{}.h5".format(NAME))


# In[8]:



#model = load_model('model.h5')
model.summary()

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "testImages2",
    labels='inferred',
    label_mode = 'categorical',
    image_size=image_size,
)

ynew = model.predict(dataset)
print(np.argmax(ynew,axis=1))

