# -*- coding: utf-8 -*-
"""LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uXthNK8sC5ImbfFPaCAGAj50VR7VhVam
"""

from google.colab import drive
import pandas as pd
import torch
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import io
import re
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Loading CSV
drive.mount('/content/drive')

df_train = pd.read_csv('/content/drive/My Drive/MLProject/train.csv') 
df_test = pd.read_csv('/content/drive/My Drive/MLProject/test.csv')

# Examine data set
print(df_train.head())
print(df_train['full_text'][2])

# Data pre-processing
# remove '\n \r \w'
df_train['full_text'] = df_train["full_text"].replace(re.compile(r'[\n\r\t]'), ' ', regex=True)
df_test['full_text'] = df_test["full_text"].replace(re.compile(r'[\n\r\t]'), ' ', regex=True)
df_train['full_text'] = df_train["full_text"].replace(re.compile(r'[^\w]'), ' ', regex=True)
df_test['full_text'] = df_test["full_text"].replace(re.compile(r'[^\w]'), ' ', regex=True)

# remove stop words
nltk.download('stopwords')
stop = stopwords.words('english')
df_train['full_text'] = df_train['full_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df_test['full_text'] = df_test['full_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(df_train['full_text'][2])

# tokenize
nltk.download('punkt')

# visualize words after tokenize
train_token = df_train['full_text'].apply(word_tokenize)
test_token = df_test['full_text'].apply(word_tokenize)

print(train_token[0])

# assign x and y
X = df_train['full_text']
y = df_train[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']]

#split train test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) 

y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()

print(X_train[0])
print(X_train.shape)
print(y_train.shape)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

train_seq = tokenizer.texts_to_sequences(X_train)
pad_train = pad_sequences(train_seq, maxlen=1250, truncating='post')

val_seq = tokenizer.texts_to_sequences(X_val)
pad_val = pad_sequences(val_seq, maxlen=1250, truncating='post')

test_seq = tokenizer.texts_to_sequences(X_test)
pad_test = pad_sequences(test_seq, maxlen=1250, truncating='post') #max length of word is 1250

print(pad_train[3]) #pad_train is a numpy array
print(pad_train.shape)

## for y, what should be the shape (6 lists each with its output? or m lists each with 6 outputs)

word_idx_count = len(word_index)
print(word_idx_count)

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

def data_sequence(input, label):
  inout_seq = []
  for i in range(input.shape[0]):
    seq = torch.Tensor(input[i,:])
    target = torch.Tensor(label[i,:])
    inout_seq.append((seq,target))
  return inout_seq

inout_seq = data_sequence(pad_train,y_train)
print(inout_seq[3])

class LSTM_Model(nn.Module):
  def __init__(self):
    super(LSTM_Model, self).__init__()
    self.embeddings = nn.Embedding(word_idx_count+1, 64)
    self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True)
    self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, bidirectiona=True)
    self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, bidirectiona=True)
    self.fc1 = torch.nnLinear(32,2)
    self.relu = torch.nnReLU()
    self.fc2 = torch.nn.Linear(2,1)

  def forward(self,inputs):
    embeds = self.embeddings(inputs)
    h1 = self.lstm1(embeds)
    h2 = self.lstm2(h1)  
    h3 = self.lstm(h2)
    h4 = self.fc1(h3)
    h4 = self.relu(h4)
    out1 = self.fc2(h4)
    out2 = self.fc2(h4)
    out3 = self.fc2(h4)
    out4 = self.fc2(h4)
    out5 = self.fc2(h4)
    out6 = self.fc2(h4)

for i in range(epochs):
  for seq, labels in inout_seq: