import json
import pandas as pd
import openpyxl


import tensorflow as tf
# helps in text preprocessing
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# helps in model building
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SpatialDropout1D
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.callbacks import EarlyStopping

import torch
import torch.nn as nn

from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

# specify GPU
device = torch.device("cuda")

# split data into train and test set
from sklearn.model_selection import train_test_split
# split data into train and test set
from sklearn.model_selection import train_test_split

import os
import numpy as np
from numpy import random
import gensim
import nltk, re
import time
import torch
import torch.nn as nn

from os import path
from sklearn import preprocessing
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, multilabel_confusion_matrix, precision_score, f1_score, recall_score, log_loss
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt

nltk.download ('wordnet')
nltk.download("punkt")
nltk.download("stopwords")

py_lemmatizer = WordNetLemmatizer ()
stop = stopwords.words("english")
exclude = []
stop_words = [word for word in stop if word not in exclude]

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_<.*?>]')

def model_plot(estimator, x_train, y_train, train_sizes=np.linspace(.1, 1.0, 5) ):
  plt.figure()
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  train_sizes, train_scores, test_scores = learning_curve(
      estimator, x_train, y_train, train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  plt.grid()
  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
  plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Test score")

  plt.legend(loc="best")
  return plt

def process_text(text: list) -> list:
  final_list = []
  alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
  for sentence in text:

    # Check if the sentence is a missing value
    if isinstance(sentence, str) == False:
        sentence = ""

    filtered_sentence = []
    
    # Lowercase
    sentence = sentence.lower()
    # Remove leading/trailing whitespace
    sentence = sentence.strip()
    # Remove extra space and tabs
    sentence = re.sub("\s+", " ", sentence)
    # Remove HTML tags/markups:
    sentence = REPLACE_BY_SPACE_RE.sub(' ', sentence) # replace REPLACE_BY_SPACE_RE symbols by space in sentence
    sentence = BAD_SYMBOLS_RE.sub('', sentence) # delete symbols which are in BAD_SYMBOLS_RE from text

    for word in word_tokenize(sentence):
      # Check if it is not numeric and its length>2 and not in stop words
      k = 1
      for letter in word:
        if letter not in alphabet:
          k = 0
      
      if (not word.isnumeric()) and (len(word) > 2) and (word not in stop_words) and (k == 1):
        # lemmatize and add to filtered list
        filtered_sentence.append(py_lemmatizer.lemmatize (word))

      string = " ".join(filtered_sentence)  # final string of cleaned words

    final_list.append(string)

  return final_list



def get_subjects(entries: list) -> list:
    subjects = []
    for entry in entries:
      subject = []
      if 'mech' in entry:
        subject.insert(-1, 'mechanical')
      if 'aero' in entry:
        subject.insert(-1, 'aeronautical')
      if 'material' in entry:
        subject.insert(-1, 'material')
      #if it mentions mechanical and aeronautical it is most likely just aeronautical.
      if subject == ['aeronautical', 'mechanical']:
        subject = ['aeronautical']
      subjects.append(subject)
    return subjects


if __name__ == "__main__":
  """
  with open('scraping_scripts/lexicon.json', 'r') as f:
    data = json.load(f)
  df_lexicon = pd.DataFrame(data)

  path1 = "Data/Scopus_by_relevance.csv"
  path2 = "Data/Scopus_by_highest_citation_first.csv"
  
  df_relevance = pd.read_csv(path1)
  df_citation = pd.read_csv(path2)

  df_data = pd.concat([df_relevance, df_citation]).drop_duplicates(keep='first')
  

  print("Processing the text fields...")
  del df_data['Cited by']
  del df_data['Link']
  del df_data['Year']

  for column in df_data.columns:
    df_data[column] = process_text(df_data[column].values.flatten().tolist())

  df_data["combined"] = df_data["Title"] + " " + df_data["Abstract"] + " " + df_data["Author Keywords"] + " " + df_data["Index Keywords"]
  df_data["subjects"] = get_subjects(df_data["combined"].values.flatten().tolist())

  remove_words = ['tissue']
  for string in remove_words:
    df_data.drop(df_data[df_data['combined'].str.contains(string)].index, inplace = True)

  df_data = df_data[df_data['subjects'].map(len) < 2]
  df_data['subjects'] = [','.join(map(str, l)) for l in df_data['subjects']]
  
  df_mech = df_data[(df_data["subjects"] == 'mechanical')].iloc[0:360, :]
  df_mat = df_data[(df_data["subjects"] == 'material')].iloc[0:360, :]
  df_aero = df_data[(df_data["subjects"] == 'aeronautical')].iloc[0:360, :]

  path = "processedstep2.xlsx"
  df_data.to_excel(path, sheet_name="Data", index=False)

  df_balanced_data = pd.concat([df_mech, df_mat, df_aero])

  del df_balanced_data['Source title']
  del df_balanced_data['Author Keywords']
  del df_balanced_data['Index Keywords']
  del df_balanced_data['combined'] 
  df_balanced_data['text'] = df_balanced_data['Title'] + " " + df_balanced_data['Abstract']
  del df_balanced_data['Title']
  del df_balanced_data['Abstract']
  """
  # Write data to excel output file
  path = "processed.xlsx"
  #df_balanced_data.to_excel(path, sheet_name="Data", index=False)
  df_balanced_data = pd.read_excel(path)

  # Show class distribution and word distribution
  labels = ['mechanical', 'aeronautical', 'material']
  plt.figure(figsize=(10,4))
  plt.ylabel("Occurence")
  df_balanced_data.subjects.value_counts().plot(kind='bar')
  

  words = []
  
  entries = df_balanced_data['text'].values.flatten().tolist()
  for entry in entries:
    sentence = entry.split()
    for word in sentence:
      words.insert(0, word)
  words = pd.Series(words)
  plt.figure(figsize=(10,4))
  plt.ylabel("Occurence")
  words.value_counts()[:40].plot(kind='bar')
  

  # Shuffle dataframe entries randomly
  df_balanced_data.sample(frac = 1)

  df_mech = df_balanced_data[(df_balanced_data["subjects"] == 'mechanical')]
  df_mat = df_balanced_data[(df_balanced_data["subjects"] == 'material')]
  df_aero = df_balanced_data[(df_balanced_data["subjects"] == 'aeronautical')]

  df_full = pd.concat([df_mech, df_mat, df_aero])
  df_train = pd.concat([df_mech.iloc[0:287], df_mat.iloc[0:287], df_aero.iloc[0:287]])
  df_val = pd.concat([df_mech.iloc[287:324], df_mat.iloc[287:324], df_aero.iloc[287:324]])
  df_trainval = pd.concat([df_mech.iloc[0:324], df_mat.iloc[0:324], df_aero.iloc[0:324]])
  df_test = pd.concat([df_mech.iloc[324:360], df_mat.iloc[324:360], df_aero.iloc[324:360]])
  df_metrics = pd.DataFrame({'Model': [], 'Parameter/Metric': [], 'Value':[]})

  """Process for model trainings, train data and plot learning curve, save hyperparametrs in a table, plot classifcation matrix (multiclass and binary), save evaluation metrics in table"""
  
  # Train & Test Naive Bayes
  x = df_full.text
  y = df_full.subjects
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state = 42)

  """nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
  
  nb.fit(x_train, y_train)

  y_pred = nb.predict(x_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=['mechanical','material','aeronautical']))
  
  model_plot(nb, x_train, y_train)

  # Train & Test SVM
  sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
  sgd.fit(x_train, y_train)

  y_pred = sgd.predict(x_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=['mechanical','material','aeronautical']))

  model_plot(sgd, x_train, y_train)

  # Train & Test KNN
  k_values = [i for i in range (1,900)]
  scores = []
  
  for k in k_values:
    
    knn = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors = k)),
        ])
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(np.mean(score))
  plt.figure()
  plt.plot(k_values, scores, marker = 'o')
  plt.xlabel("K Values")
  plt.ylabel("Accuracy Score")

  idx = scores.index(max(scores))
  k = k_values[idx]

  knn = Pipeline([
      ('vect', CountVectorizer()),
      ('tfidf', TfidfTransformer()),
      ('clf', KNeighborsClassifier(n_neighbors=k)),
      ])
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)

  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  print(classification_report(y_test, y_pred,target_names=['mechanical','material','aeronautical']))

  model_plot(knn, x_train, y_train)

  # Train & Test RNN
  # The maximum number of words to be used. (most frequent)
  MAX_NB_WORDS = 100000
  # Max number of words in each entry.
  MAX_SEQUENCE_LENGTH = 750
  # This is fixed.
  EMBEDDING_DIM = 100
  
  tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
  tokenizer.fit_on_texts(df_full.text)
  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))

  x = tokenizer.texts_to_sequences(df_full.text)
  x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)

  y = pd.get_dummies(df_full.subjects).values
  
  x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.10, random_state = 42)

  rnn = Sequential()
  rnn.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x.shape[1]))
  rnn.add(SpatialDropout1D(0.2))
  rnn.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
  rnn.add(Dense(3, activation='softmax'))
  rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  epochs = 50
  batch_size = 64

  history = rnn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.111111,callbacks=[EarlyStopping(monitor='val_loss', patience=15, min_delta=0.0001)])
  hist_dict = history.history
  print(hist_dict.keys())
  accr = rnn.evaluate(x_test,y_test)
  print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

  plt.figure()
  plt.title('Loss')
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  
  plt.figure()
  plt.title('Accuracy')
  plt.plot(history.history['accuracy'], label='train')
  plt.plot(history.history['val_accuracy'], label='test')
  plt.legend()
  """

  # Train & Test BERT


  # Compare Metrics (Calculate overall score based on metric priorities)


  plt.show()