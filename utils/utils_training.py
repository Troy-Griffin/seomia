import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def rnn_train(x_train, y_train, x, max_words, embedding_dim, epochs=4, batch_size=64):
    rnn = Sequential()
    rnn.add(Embedding(max_words, embedding_dim, input_length=x.shape[1]))
    rnn.add(SpatialDropout1D(0.2))
    rnn.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    rnn.add(Dense(3, activation="softmax"))
    rnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = rnn.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.111111,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, min_delta=0.0001)],
    )

    return rnn, history


def svm_train(x_train, y_train):
    svm = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                SGDClassifier(
                    loss="hinge",
                    penalty="l2",
                    alpha=1e-3,
                    random_state=42,
                    max_iter=5,
                    tol=None,
                ),
            ),
        ]
    )
    svm.fit(x_train, y_train)
    return svm


def mnb_train(x_train, y_train):
    mnb = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )
    mnb.fit(x_train, y_train)
    return mnb


def knn_train(x_train, y_train, k):
    knn = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    knn.fit(x_train, y_train)
    return knn


def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(
        shape=(max_len,), dtype=tf.int32, name="input_word_ids"
    )
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation="relu")(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation="relu")(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(3, activation="softmax")(net)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_mask, segment_ids], outputs=out
    )
    model.compile(
        tf.keras.optimizers.Adam(lr=10000),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


class BERTArch(nn.Module):
    def __init__(self, bert):
        super(BERTArch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 3)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x


def train(model, train_dataloader, cross_entropy, optimizer):
    model.train()

    total_loss, _ = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        # progress update after every 50 batches.
        if step % 50 == 0 and step != 0:
            print("  Batch {:>5,}  of  {:>5,}.".format(step, len(train_dataloader)))

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(
            sent_id,
            mask,
        )

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        preds = preds * (-1)
        predict = np.argmin(preds, axis=1)

        accuracy = accuracy_score(labels, predict)

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds, accuracy


def evaluate(model, val_dataloader, cross_entropy):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, _ = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):
        # Progress update every 50 batches.
        if step % 50 == 0 and step != 0:
            print("  Batch {:>5,}  of  {:>5,}.".format(step, len(val_dataloader)))

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            predict = np.argmax(preds, axis=1)

            accuracy = accuracy_score(labels, predict)

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, accuracy
