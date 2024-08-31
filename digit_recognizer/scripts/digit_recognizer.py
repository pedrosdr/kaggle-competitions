import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import MaxPool2D, Input, Conv2D, AveragePooling2D
from sklearn.model_selection import train_test_split

base = pd.read_csv('data/train.csv')
base = base.to_numpy()

x = base[:,1:]
y = base[:,0:1]

x = x.reshape(len(x), 28, 28, 1)
plt.imshow(x[232])

xtrain, ytrain, xtest, ytest = train_test_split(x, y)

model1 = Sequential([
    Input([28,28,1]),
    AveragePooling2D([2,2])
])

model2 = Sequential([
    Input([28,28,1]),
    MaxPool2D([2,2])
])

imagens_tratadas = model1.predict(x)
plt.imshow(imagens_tratadas[232])

imagens_tratadas = model2.predict(x)
plt.imshow(imagens_tratadas[232])
