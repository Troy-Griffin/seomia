# import tokenization_bert
import os
import time
from os import path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import torch
import torch.nn as nn
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from utils.utils_parsing import get_metrics, get_subjects, process_text
from utils.utils_plot import (
    model_plot,
    plot_knn,
    plot_matrix,
    plot_matrix_from_pred,
    plot_nn_curves,
    plot_words_and_classes,
)
from utils.utils_tokenization import bert_encode, bert_preprocess
from utils.utils_training import (
    BERTArch,
    build_model,
    evaluate,
    knn_train,
    mnb_train,
    rnn_train,
    svm_train,
    train,
)

# from tqdm.notebook import tqdm


if __name__ == "__main__":
    path1 = "Data/Scopus_by_relevance.csv"
    path2 = "Data/Scopus_by_highest_citation_first.csv"

    df_relevance = pd.read_csv(path1)
    df_citation = pd.read_csv(path2)

    df_data = pd.concat([df_relevance, df_citation]).drop_duplicates(keep="first")

    print("Processing the text fields...")
    del df_data["Cited by"]
    del df_data["Link"]
    del df_data["Year"]

    for column in df_data.columns:
        df_data[column] = process_text(df_data[column].values.flatten().tolist())

    df_data["combined"] = (
        df_data["Title"]
        + " "
        + df_data["Abstract"]
        + " "
        + df_data["Author Keywords"]
        + " "
        + df_data["Index Keywords"]
    )
    df_data["subjects"] = get_subjects(df_data["combined"].values.flatten().tolist())

    df_data = df_data[df_data["subjects"].map(len) < 2]
    df_data["subjects"] = [",".join(map(str, l)) for l in df_data["subjects"]]

    # Save output here to show all single subject entries
    filepath = "processedstep1.xlsx"
    df_data.to_excel(filepath, sheet_name="Data", index=False)
    # df_data = pd.read_excel(filepath)

    # Show class distribution and word distribution before balancing
    df_data["text"] = df_data["Title"] + " " + df_data["Abstract"]
    plot_words_and_classes(df_data)

    remove_entries = ["tissue"]
    for string in remove_entries:
        df_data.drop(
            df_data[df_data["combined"].str.contains(string)].index, inplace=True
        )

    df_aero = df_data[(df_data["subjects"] == "aeronautical")]
    length = len(df_aero)
    df_mech = df_data[(df_data["subjects"] == "mechanical")].iloc[0:length, :]
    df_mat = df_data[(df_data["subjects"] == "material")].iloc[0:length, :]
    df_data = pd.concat([df_mat, df_mech, df_aero])

    bias_words = ["engineering", "material", "mechanical", "aeronautical"]

    for string in bias_words:
        df_data["Title"] = df_data["Title"].replace(string, "", regex=True)
        df_data["Abstract"] = df_data["Abstract"].replace(string, "", regex=True)

    filepath = "processedstep2.xlsx"
    df_data.to_excel(filepath, sheet_name="Data", index=False)

    del df_data["Source title"]
    del df_data["Author Keywords"]
    del df_data["Index Keywords"]
    del df_data["combined"]
    df_data["text"] = df_data["Title"] + " " + df_data["Abstract"]
    del df_data["Title"]
    del df_data["Abstract"]

    # Write data to excel output file
    filepath = "processedstep3.xlsx"
    # df_data.to_excel(filepath, sheet_name="Data", index=False)
    # df_data = pd.read_excel(filepath)

    plot_words_and_classes(df_data)

    # Shuffle dataframe entries randomly
    df_data.sample(frac=1)

    df_mech = df_data[(df_data["subjects"] == "mechanical")]
    df_mat = df_data[(df_data["subjects"] == "material")]
    df_aero = df_data[(df_data["subjects"] == "aeronautical")]

    df_full = pd.concat([df_mech, df_mat, df_aero])
    df_metrics = pd.DataFrame(
        {"Model": [], "Class": [], "Parameter/Metric": [], "Value": []}
    )

    # Train & Test Naive Bayes
    t0 = time.time()
    x = df_full.text
    y = df_full.subjects
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    mnb = mnb_train(x_train, y_train)

    y_pred = mnb.predict(x_test)

    t1 = time.time()
    t_train = t1 - t0

    print("accuracy %s" % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))

    report = classification_report(
        y_test,
        y_pred,
        target_names=["mechanical", "material", "aeronautical"],
        output_dict=True,
    )
    df_metrics = get_metrics(df_metrics, report, "MNB", t_train)

    model_plot(mnb, x_train, y_train)

    plot_matrix("MNB", mnb, x_test, y_test)

    # Train & Test SVM
    t0 = time.time()
    svm = svm_train(x_train, y_train)

    y_pred = svm.predict(x_test)

    t1 = time.time()
    t_train = t1 - t0

    print("accuracy %s" % accuracy_score(y_pred, y_test))
    print(
        classification_report(
            y_test, y_pred, target_names=["mechanical", "material", "aeronautical"]
        )
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["mechanical", "material", "aeronautical"],
        output_dict=True,
    )
    df_metrics = get_metrics(df_metrics, report, "SVM", t_train)

    model_plot(svm, x_train, y_train)

    plot_matrix("SVM", svm, x_test, y_test)

    # Train & Test KNN
    t0 = time.time()
    k_values = [i for i in range(1, 900)]
    scores = []

    for k in k_values:
        knn = knn_train(x_train, y_train, k)

        y_pred = knn.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(np.mean(score))

    plot_knn(k_values, scores)
    idx = scores.index(max(scores))
    k = k_values[idx]

    knn = knn_train(x_train, y_train, k)

    y_pred = knn.predict(x_test)

    t1 = time.time()
    t_train = t1 - t0
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print(
        classification_report(
            y_test, y_pred, target_names=["mechanical", "material", "aeronautical"]
        )
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["mechanical", "material", "aeronautical"],
        output_dict=True,
    )
    df_metrics = get_metrics(df_metrics, report, "KNN", t_train)

    model_plot(knn, x_train, y_train)

    plot_matrix("KNN", knn, x_test, y_test)

    # Train & Test RNN
    t0 = time.time()
    # The maximum number of words to be used. (most frequent)
    max_words = 100000
    # Max number of words in each entry.
    max_seq_length = 750
    # This is fixed.
    embedding_dim = 100

    tokenizer = Tokenizer(
        num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True
    )
    tokenizer.fit_on_texts(df_full.text)
    word_index = tokenizer.word_index

    x = tokenizer.texts_to_sequences(df_full.text)
    x = pad_sequences(x, maxlen=max_seq_length)

    y = pd.get_dummies(df_full.subjects).values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.10, random_state=42
    )

    rnn, history = rnn_train(x_train, y_train, x, max_words, embedding_dim)

    t1 = time.time()
    t_train = t1 - t0
    accr = rnn.evaluate(x_test, y_test)

    print("Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))
    y_pred = rnn.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print(
        classification_report(
            y_test, y_pred, target_names=["mechanical", "material", "aeronautical"]
        )
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["mechanical", "material", "aeronautical"],
        output_dict=True,
    )
    df_metrics = get_metrics(df_metrics, report, "RNN", t_train)

    plot_nn_curves("RNN", history)

    plot_matrix_from_pred("RNN", y_test, y_pred)

    # Train & Test BERT
    t0 = time.time()

    possible_labels = df_full.subjects.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df = df_full

    df["subjects"] = df.subjects.replace(label_dict)

    train_text, temp_text, train_labels, temp_labels = train_test_split(
        df_full["text"],
        df_full["subjects"],
        random_state=2018,
        test_size=0.2,
        stratify=df_full["subjects"],
    )

    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels
    )

    bert, tokens_train, tokens_val, tokens_test = bert_preprocess(
        train_text, val_text, test_text
    )

    train_seq = torch.tensor(tokens_train["input_ids"])
    train_mask = torch.tensor(tokens_train["attention_mask"])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val["input_ids"])
    val_mask = torch.tensor(tokens_val["attention_mask"])
    val_y = torch.tensor(val_labels.tolist())

    test_seq = torch.tensor(tokens_test["input_ids"])
    test_mask = torch.tensor(tokens_test["attention_mask"])
    test_y = torch.tensor(test_labels.tolist())

    # define a batch size
    batch_size = 256

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    model = BERTArch(bert)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )

    weights = torch.tensor(class_weights, dtype=torch.float)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    # number of training epochs
    epochs = 10

    # set initial loss to infinite
    best_valid_loss = float("inf")

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []
    train_accuracies = []
    val_accuracies = []

    # for each epoch
    for epoch in range(epochs):
        print("\n Epoch {:} / {:}".format(epoch + 1, epochs))

        # train model
        train_loss, _, train_accuracy = train(
            model, train_dataloader, cross_entropy, optimizer
        )

        # evaluate model
        valid_loss, _, val_accuracy = evaluate(model, val_dataloader, cross_entropy)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "saved_weights.pt")

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"\nTraining Loss: {train_loss:.3f}")
        print(f"Validation Loss: {valid_loss:.3f}")
        print(f"\nTraining accuracy: {train_accuracy:.3f}")
        print(f"Validation accuracy: {val_accuracy:.3f}")

    path = "saved_weights.pt"
    model.load_state_dict(torch.load(path))

    preds = model(test_seq, test_mask)
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)

    print(classification_report(test_y, preds))

    t1 = time.time()
    t_train = t1 - t0

    print(
        classification_report(
            test_y, preds, target_names=["mechanical", "material", "aeronautical"]
        )
    )

    report = classification_report(
        test_y,
        preds,
        target_names=["mechanical", "material", "aeronautical"],
        output_dict=True,
    )
    df_metrics = get_metrics(df_metrics, report, "BERT", t_train)

    plot_nn_curves("BERT", history)

    plot_matrix_from_pred("BERT", test_y, preds)

    # Compare Metrics (Calculate overall score based on metric priorities) will be done in excel
    filepath = "metrics.xlsx"
    df_metrics.to_excel(filepath, sheet_name="Data", index=False)

    plt.show()
