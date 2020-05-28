import pickle
import numpy as np

from pathlib import Path
from gensim.models import Word2Vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from collections import Counter


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
def filter_data(lines: list, threshold: int = 5):
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


def benchmark(classifier, X, y, k_th, splits=5):
    scores = list()

    precisions = list()
    recalls = list()

    for train, test in StratifiedKFold(n_splits=splits).split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        classifier.fit(X_train, y_train)
        y_prediction = classifier.predict(X_test)

        y_pred = list()

        for i, x_test in enumerate(X_test):
            probs = classifier.predict_proba(np.array([x_test]))
            probs_arr = [round(float(value), 2) for value in list(probs[0])]
            
            max_index = probs_arr.index(max(probs_arr))
            true_value = y_test[i]

            indexes = list(np.argpartition(probs[0], -k_th)[-k_th:])

            if true_value in indexes:
                y_pred.append(true_value)
            else:
                y_pred.append(max_index)

        y_pred = np.array(y_pred)
        precisions.append(precision_score(y_test, y_pred, average="macro", labels=np.unique(y_pred)))
        recalls.append(recall_score(y_test, y_pred, average="macro", labels=np.unique(y_pred)))

    return (sum(precisions) / len(precisions), sum(recalls) / len(recalls))


def train_evaluate_coster(file_path: str):
    print("Reading file ...")
    data = open(file_path)
    lines = read_file(data)
    data.close()
    print("Done !")

    print("Filtering the data with fqns higher than a threshold ...")
    lines_filtered = filter_data(lines)
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
        ("RF", RandomForestClassifier(n_jobs=8)),
    ]

    print("Metrics for the models ...")

    for name, classifier in models_test:
        print(name)
        print(f"Results for the classifier {name}")
        tops = [1, 3, 5]

        for top in tops:
            precision_k, recall_k = benchmark(classifier, X, y, top)
            print(f"Precision@{top}={round(precision_k, 2)} Recall@{top}={round(recall_k, 2)}")
        
        print()
    print("Done !")
