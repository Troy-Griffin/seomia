import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve


def plot_words_and_classes(df_data):
    # Show class distribution and word distribution after balancing
    plt.figure(figsize=(10, 4))
    plt.ylabel("Occurence")
    df_data.subjects.value_counts().plot(kind="bar")

    words = []
    entries = df_data["text"].values.flatten().tolist()
    for entry in entries:
        sentence = entry.split()
        for word in sentence:
            words.insert(0, word)
    words = pd.Series(words)
    plt.figure(figsize=(10, 4))
    plt.ylabel("Occurence")
    words.value_counts()[:40].plot(kind="bar")


def model_plot(estimator, x_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(f"{estimator} training curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x_train, y_train, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Test score")

    plt.legend(loc="best")
    return plt


def plot_knn(k_values, scores):
    plt.figure()
    plt.title("optimise k value")
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")


def plot_matrix(
    model_name,
    estimator,
    x_test,
    y_test,
    classes=["mechanical", "material", "aeronautical"],
):
    disp = ConfusionMatrixDisplay.from_estimator(
        estimator, x_test, y_test, display_labels=classes, cmap=plt.cm.Blues
    )
    disp.ax_.set_title(f"{model_name} matrix")
    print(disp.confusion_matrix)


def plot_matrix_from_pred(
    estimator, y_test, y_pred, classes=["mechanical", "material", "aeronautical"]
):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=classes, cmap=plt.cm.Blues
    )
    disp.ax_.set_title(f"{estimator} matrix")
    print(disp.confusion_matrix)


def plot_nn_curves(estimator, history):
    plt.figure()
    plt.title(f"{estimator} learning curve")
    plt.title("Loss")
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.legend()

    plt.figure()
    plt.title(f"{estimator} training curve")
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.legend()
