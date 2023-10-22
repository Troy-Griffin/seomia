import json
import pandas as pd
import openpyxl

import tensorflow_hub as hub
import tokenization_bert
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

from tqdm.notebook import tqdm

from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import classification_report
import transformers

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import transformers
from transformers import AutoModel, BertTokenizerFast



from sklearn.utils.class_weight import compute_class_weight

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

"""nltk.download ('wordnet')
nltk.download("punkt")
nltk.download("stopwords")"""

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

def get_metrics(df, report, model_name, t):
  row = {'Model': model_name, 'Class': 'total', 'Parameter/Metric': 'Training Time', 'Value': round(t, 2)}
  df = df._append(row, ignore_index=True)
  for label in report:
    if label == 'accuracy':
      row = {'Model': model_name, 'Class': 'total', 'Parameter/Metric': label.title(), 'Value': round(report[label], 2)}
      df = df._append(row, ignore_index=True)
      continue
    for metric in report[label]:
      if metric == 'support':
        continue
      row = {'Model': model_name, 'Class': label.title(), 'Parameter/Metric': metric.title(), 'Value': round(report[label][metric], 2)}
      df = df._append(row, ignore_index=True)
  return df

def bert_encode(texts, tokenizer, max_len=512):
  all_tokens = []
  all_masks = []
  all_segments = []
  
  for text in texts:
      text = tokenizer.tokenize(text)
          
      text = text[:max_len-2]
      input_sequence = ["[CLS]"] + text + ["[SEP]"]
      pad_len = max_len - len(input_sequence)
      
      tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
      pad_masks = [1] * len(input_sequence) + [0] * pad_len
      segment_ids = [0] * max_len
      
      all_tokens.append(tokens)
      all_masks.append(pad_masks)
      all_segments.append(segment_ids)
    
  return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512):
  input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
  segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

  pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
  clf_output = sequence_output[:, 0, :]
  net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
  net = tf.keras.layers.Dropout(0.2)(net)
  net = tf.keras.layers.Dense(32, activation='relu')(net)
  net = tf.keras.layers.Dropout(0.2)(net)
  out = tf.keras.layers.Dense(3, activation='softmax')(net)
  
  model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
  model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
  
  return model



if __name__ == "__main__":
  
  """with open('scraping_scripts/lexicon.json', 'r') as f:
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

  df_data = df_data[df_data['subjects'].map(len) < 2]
  df_data['subjects'] = [','.join(map(str, l)) for l in df_data['subjects']]
  #Save output here to show all single subject entries
  path = "processedstep1.xlsx"
  df_data.to_excel(path, sheet_name="Data", index=False)
  #df_data = pd.read_excel(path)

  # Show class distribution and word distribution before balancing
  df_data['text'] = df_data['Title'] + " " + df_data['Abstract']
  labels = ['mechanical', 'aeronautical', 'material']
  plt.figure(figsize=(10,4))
  plt.ylabel("Occurence")
  df_data.subjects.value_counts().plot(kind='bar')
  

  words = []
  entries = df_data['text'].values.flatten().tolist()
  for i, entry in enumerate(entries):
    print(i)
    sentence = entry.split()
    for word in sentence:
      words.insert(0, word)
  words = pd.Series(words)
  plt.figure(figsize=(10,4))
  plt.ylabel("Occurence")
  words.value_counts()[:40].plot(kind='bar')

  remove_entries = ['tissue']
  for string in remove_entries:
    df_data.drop(df_data[df_data['combined'].str.contains(string)].index, inplace = True)

  df_aero = df_data[(df_data["subjects"] == 'aeronautical')]
  length = len(df_aero)
  df_mech = df_data[(df_data["subjects"] == 'mechanical')].iloc[0:length,:]
  df_mat = df_data[(df_data["subjects"] == 'material')].iloc[0:length,:]
  df_data = pd.concat([df_mat, df_mech, df_aero])
  

  bias_words = ['engineering','material','mechanical','aeronautical']
  
  for string in bias_words:
    df_data['Title'] = df_data['Title'].replace(string, '', regex = True)
    df_data['Abstract'] = df_data['Abstract'].replace(string, '', regex = True)
  
  path = "processedstep2.xlsx"
  df_data.to_excel(path, sheet_name="Data", index=False)

  del df_data['Source title']
  del df_data['Author Keywords']
  del df_data['Index Keywords']
  del df_data['combined'] 
  df_data['text'] = df_data['Title'] + " " + df_data['Abstract']
  del df_data['Title']
  del df_data['Abstract']"""
  
  # Write data to excel output file
  path = "processedstep3.xlsx"
  #df_data.to_excel(path, sheet_name="Data", index=False)
  df_data = pd.read_excel(path)

  """# Show class distribution and word distribution after balancing
  labels = ['mechanical', 'aeronautical', 'material']
  plt.figure(figsize=(10,4))
  plt.ylabel("Occurence")
  df_data.subjects.value_counts().plot(kind='bar')
  

  words = []
  entries = df_data['text'].values.flatten().tolist()
  for entry in entries:
    sentence = entry.split()
    for word in sentence:
      words.insert(0, word)
  words = pd.Series(words)
  plt.figure(figsize=(10,4))
  plt.ylabel("Occurence")
  words.value_counts()[:40].plot(kind='bar')"""
  

  # Shuffle dataframe entries randomly
  df_data.sample(frac = 1)

  df_mech = df_data[(df_data["subjects"] == 'mechanical')]
  df_mat = df_data[(df_data["subjects"] == 'material')]
  df_aero = df_data[(df_data["subjects"] == 'aeronautical')]

  df_full = pd.concat([df_mech, df_mat, df_aero])
  df_metrics = pd.DataFrame({'Model': [],'Class': [], 'Parameter/Metric': [], 'Value':[]})

  """Process for model trainings, train data and plot learning curve, save hyperparametrs in a table, plot classifcation matrix (multiclass and binary), save evaluation metrics in table"""
  
  # Train & Test Naive Bayes
  t0 = time.time()
  x = df_full.text
  y = df_full.subjects
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state = 42)

  nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
  
  nb.fit(x_train, y_train)

  y_pred = nb.predict(x_test)

  t1 = time.time()

  t_train = t1 - t0

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred))
  report = classification_report(y_test, y_pred,target_names=['mechanical','material','aeronautical'], output_dict=True)
  
  df_metrics = get_metrics(df_metrics, report, 'MNB', t_train)
  print(df_metrics)

  model_plot(nb, x_train, y_train)


  """
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
  rnn.compile(loss='categorical_crossentropy', optimizer='adamw', metrics=['accuracy'])

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
  

  # Train & Test BERT
  possible_labels = df_full.subjects.unique()

  label_dict = {}
  for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
  df = df_full
  df['subjects'] = df.subjects.replace(label_dict)

  x_train, x_temp, y_train, y_temp = train_test_split(df.text, df.subjects, random_state=42, test_size=0.2)

  x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, random_state=42, test_size=0.5)
  
  module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
  bert_layer = hub.KerasLayer(module_url, trainable=True)

  vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
  do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
  tokenizer = tokenization_bert.FullTokenizer(vocab_file, do_lower_case)

  max_len = 512
  train_input = bert_encode(x_train, tokenizer, max_len=max_len)
  test_input = bert_encode(x_test, tokenizer, max_len=max_len)
  train_labels = tf.keras.utils.to_categorical(y_train, num_classes=3)
  test_labels = tf.keras.utils.to_categorical(y_test, num_classes=3)
  
  model = build_model(bert_layer, max_len=max_len)
  model.summary()
  

  checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
  earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, verbose=1)

  epochs = 50
  batch_size = 16

  train_history = model.fit(train_input, train_labels,epochs = epochs, batch_size=batch_size, validation_split=0.1111111,callbacks=[checkpoint, earlystopping],verbose=1)
  
  #Predictions for the test data:

  model.load_weights('model.h5')

  test_pred = model.predict(test_input)

  test_pred = np.argmax(test_pred, axis=1)

  print(classification_report(y_test, test_pred,target_names=['mechanical','material','aeronautical']))
  plt.figure()
  plt.title('Loss')
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  
  plt.figure()
  plt.title('Accuracy')
  plt.plot(train_history.history['accuracy'], label='train')
  plt.plot(train_history.history['val_accuracy'], label='test')
  plt.legend()
  """
  # Compare Metrics (Calculate overall score based on metric priorities)
  plt.show()