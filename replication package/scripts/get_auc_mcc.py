import numpy as np

from tabulate import tabulate
from gensim.models import Word2Vec
from sklearn.metrics import auc, roc_curve
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
    coeff_metrics = dict()

    for k, (train, test) in enumerate(StratifiedKFold(n_splits=10).split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        print(f"Fold # {k + 1}")

        print("Training ...")
        classifier.fit(X_train, y_train)
        print("Done!")

        y_score = classifier.predict_proba(X[test])
        y_score2 = classifier.predict(X[test])

        y_test_dummies = pd.get_dummies(y[test], drop_first=False).values
        for i in range(n_classes):
            y_score_transformed = []

            for value in y_score[:, i]:
                if value >= 0.85:
                    y_score_transformed.append(1)
                elif 0.5 <= value < 0.85:
                    y_score_transformed.append(0)
                else:
                    y_score_transformed.append(-1)

            y_score_transformed = np.array(y_score_transformed)
            y_dummies_transformed = [1 if value == 1 else -1 for value in y_test_dummies[:, i]]
            y_matthew_metric = matthews_corrcoef(y_dummies_transformed, y_score_transformed)

            if i in list(coeff_metrics.keys()):
                coeff_metrics[i].append(y_matthew_metric)
            else:
                coeff_metrics[i] = [y_matthew_metric]
    
    mean_coeff = list()
    for i in range(n_classes):
        mean_coeff.append(np.array(coeff_metrics[i]).mean())
    
    return mean_coeff



def tolerant_mean(arrs):
    list_arrays = [list(np_array) for np_array in arrs]
    lens = [len(i) for i in list_arrays]
    arr = np.ma.empty((np.max(lens), len(list_arrays)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def auc_roc_curves(classifier, X, y, n_classes):
    tprs_fold = dict()
    fprs_fold = dict()
    aucs_fold = dict()

    cv = StratifiedKFold(n_splits=10)

    for i, (train, test) in enumerate(cv.split(X, y)):
        print(f"Fold # {i + 1}")

        print("Training ...")
        classifier.fit(X[train], y[train])
        print("Done!")

        y_score = classifier.predict_proba(X[test])

        y_test = pd.get_dummies(y[test], drop_first=False).values
        for j in range(n_classes):

            fpr_metrics, tpr_metrics, _ = roc_curve(y_test[:, j], y_score[:, j])
            roc_auc_metrics = auc(fpr_metrics, tpr_metrics)

            if j in list(tprs_fold.keys()):
                tprs_fold[j].append(tpr_metrics)
                fprs_fold[j].append(fpr_metrics)
                aucs_fold[j].append(roc_auc_metrics)
            else:
                tprs_fold[j] = [tpr_metrics]
                fprs_fold[j] = [fpr_metrics]
                aucs_fold[j] = [roc_auc_metrics]

    mean_folds_tprs = list()
    mean_folds_fprs = list()
    mean_folds_aucs = list()

    for i in range(n_classes):
        mean_arrays_tprs, _ = tolerant_mean(tprs_fold[i])
        mean_arrays_fprs, _ = tolerant_mean(fprs_fold[i])
        
        mean_folds_tprs.append(mean_arrays_tprs)
        mean_folds_fprs.append(mean_arrays_fprs)
        mean_folds_aucs.append(np.array(aucs_fold[i]).mean())

    return mean_folds_aucs



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
    
    auc_table = list()

    print("Calculating AUC-ROC values ...")
    for name, classifier in models_test:
        print(f"Calculating AUC-ROC values for classifier {name}")
        auc_row = list()
        auc_row.append(name)

        auc_values = auc_roc_curves(classifier, X, y, len(libs_mapping))
        auc_row.append(np.mean(auc_values))

        auc_table.append(auc_row)
    print("Done !")
        
    print()
    print("Mean of AUC values per classifier")
    print(tabulate(auc_table))

    mcc_table = list()

    print("Calculating MCC values ...")
    for name, classifier in models_test:
        print(f"Calculating AUC-ROC values for classifier {name}")
        mcc_row = list()
        mcc_row.append(name)

        matthews_values = matthews_coeff_classes(classifier, X, y, len(libs_mapping))
        mcc_row.append(np.mean(matthews_values))

        mcc_table.append(mcc_row)
    print("Done !")

    print()
    print("Mean of MCC values per classifier")
    print(tabulate(auc_table))
