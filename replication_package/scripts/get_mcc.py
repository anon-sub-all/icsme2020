import numpy as np
import pandas as pd

from tabulate import tabulate
from gensim.models import Word2Vec
from sklearn.metrics import auc, roc_curve, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# Classifiers
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def read_file(file_handler):
    lines = list()

    while True:
        line = file_handler.readline()
        if not line:
            break
        else:
            line = line.strip()
            lines.append(line)
    return lines


def obtain_data(lines: list):
    sentences = list()

    for line in lines:
        line_splitted = line.split()
        line_processed = line_splitted[:-1]
        line_processed_lower = list(map(lambda word: word.lower(), line_processed))

        sentences.append(line_processed_lower)
    return sentences


# Returns those FQNs with an ocurrence higher than threshold
def filter_data(lines: list, threshold: int):
    counter_data_dict = dict()

    for line in lines:
        line_splitted = line.split()
        fqn = line_splitted[-1]
        
        if fqn in counter_data_dict:
            counter_data_dict[fqn] += 1
        else:
            counter_data_dict[fqn] = 1
    
    fqns_filtered = [fqn for fqn, presence in counter_data_dict.items() if presence >= threshold]
    lines_filtered = list()

    for line in lines:
        line_splitted = line.split()
        fqn = line_splitted[-1]

        if fqn in fqns_filtered:
            lines_filtered.append(line)
    
    return lines_filtered


def get_vectors(model, lines):
    inputs = list()
    output = list()

    classes_numbered = dict()
    k = 0

    for line in lines:
        line_splitted = line.split()
        corpus = line_splitted[:-1]
        clazz = line_splitted[-1]

        vectors = [model.wv[word.lower()] for word in corpus]
        line_vector = sum(vectors) / len(vectors)
        inputs.append(line_vector)

        if clazz in classes_numbered:
            output.append(classes_numbered[clazz])
        else:
            classes_numbered[clazz] = k
            output.append(k)
            k += 1

    return np.array(inputs), np.array(output), classes_numbered


def matthews_coeff_classes(classifier, X, y, n_classes):
    mean_coeff = list()

    for k, (train, test) in enumerate(StratifiedKFold(n_splits=10).split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        print(f"Fold # {k + 1}")

        print("Training ...")
        classifier.fit(X_train, y_train)
        print("Done!")

        y_pred = classifier.predict(X[test])
        mean_coeff.append(matthews_corrcoef(y_test, y_pred))
    
    return mean_coeff


def get_metrics():
    print("Reading file ...")
    data = open(f"data/all_snippets.txt")
    lines = read_file(data)
    data.close()
    print("Done !")

    print("Filtering the data with fqns higher than a threshold ...")
    lines_filtered = filter_data(lines, 10)
    print("Done !")

    print("Getting the corpus of the data to vectorize it ...")
    sentences2vec = obtain_data(lines_filtered)
    print("Done !")

    print("Vectorizing the corpus of the data ...")
    model_w2vec = Word2Vec(sentences2vec, min_count=1)
    print("Done !")

    print("Obtaining attributes and output for the data ...")
    X, y, libs_mapping = get_vectors(model_w2vec, lines_filtered)
    print("Done !")

    models_test = [
        ("DT", DecisionTreeClassifier()),
        ("GNB", GaussianNB()),
        ("BNB", BernoulliNB()),
        ("RF", RandomForestClassifier(n_jobs=8)),
        ("KNN", KNeighborsClassifier(n_jobs=8)),
    ]

    mcc_table = list()

    print("Calculating MCC values ...")
    for name, classifier in models_test:
        print(f"Calculating MCC values for classifier {name}")
        mcc_row = list()
        mcc_row.append(name)

        matthews_values = matthews_coeff_classes(classifier, X, y, len(libs_mapping))
        mcc_row.append(np.mean(matthews_values))

        mcc_table.append(mcc_row)
    print("Done !")

    print()
    print("Mean of MCC values per classifier")
    print(tabulate(mcc_table))
