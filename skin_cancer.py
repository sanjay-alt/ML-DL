# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:43:53 2021

@author: NEW
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_data = pd.read_csv('hmnist_28_28_RGB.csv')
df_data.head()

X = df_data.iloc[:,:2352]
Y = df_data['label']
print(X.shape)
print(Y.shape)

df_metadata = pd.read_csv('HAM10000_metadata.csv')


sns.countplot(x = 'dx', data = df_metadata)
plt.xlabel('Disease', size=12)
plt.ylabel('Frequency', size=12)
plt.title('Frequency Distribution of Classes', size=16)


from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler()
x,y  = oversample.fit_resample(X,Y)

x = np.array(x).reshape(-1,28,28,3)
print(x.shape)
print(y.shape)
#normalising data
x = x/255

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

#building the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D

model = Sequential()
model.add(Conv2D(16, kernel_size = (3,3), input_shape = (28, 28, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = 'adam',
              metrics = ['accuracy'])

history = model.fit(x_train,y_train,validation_split=0.2,batch_size = 128,epochs = 20)

loss, acc = model.evaluate(x_test, y_test, verbose=2)

predictions = model.predict(x_test)
final_predictions = []
for i in range(predictions.shape[0]):
    final_predictions.append(np.argmax(predictions[i]))
print("{} {}".format(final_predictions,y_test))


saved_model_dir = '' #means current directory
tf.saved_model.save(model, saved_model_dir) #saves to the current directory

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) 
tflite_model = converter.convert() #converts our model into a .tflite model which flutter uses for ondevice machine learning

with open('skin_lads.tflite', 'wb') as f: #to write the converted model into a file, written as binary so add 'wb' instead of 'w'
    f.write(tflite_model)