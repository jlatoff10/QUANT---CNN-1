
import numpy as np

import pandas_ta as pta
import pandas as pd

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
#%%
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
#%%
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