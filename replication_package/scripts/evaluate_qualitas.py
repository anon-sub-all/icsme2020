import pickle
import numpy as np

from pathlib import Path
from collections import Counter
from sklearn.metrics import precision_score, recall_score


def get_predictions(vector, model, mapping, k_th):
	probs = model.predict_proba(np.array([vector]))
	probs_arr = [round(float(value), 2) for value in list(probs[0])]

	max_index = probs_arr.index(max(probs_arr))
	indexes = list(np.argpartition(probs[0], -k_th)[-k_th:])

	predictions = list()

	for key, value in libs_mapping.items():
	    if value in indexes:
	        predictions.append(key)

	return predictions, max_index


def benchmark(data, trained_model, w2vec_model, libs_mapping, k_th):
	precisions = list()
	recalls = list()
	y_test = list()
	y_pred = list()

	for i, data_line in enumerate(data):

		if i % 5000 == 0:
			print(f"Processed {i} lines ...")

		fqn = data_line[-1]
		tokens = data_line[:-1]
		vectors = list()
		absences = 0

		for token in tokens:
			token_lower = token.lower()
			if token_lower in list(w2vec_model.wv.vocab.keys()):
			    vectors.append(w2vec_model.wv[token_lower])
			else:
				absences += 1

		vector_single = w2vec_model.wv[tokens[0].lower()]
		vector_model_code = sum(vectors) / len(vectors)

		predictions_single, max_single = get_predictions(vector_single, trained_model, libs_mapping, k_th)
		predictions_vector, max_vector = get_predictions(vector_model_code, trained_model, libs_mapping, k_th)

		true_value = libs_mapping[fqn]

		y_test.append(true_value)

		if fqn in predictions_vector or fqn in predictions_single:
		    y_pred.append(true_value)
		else:
			y_pred.append(max(max_single, max_vector))

	y_test = np.array(y_test)
	y_pred = np.array(y_pred)

	return (precision_score(y_test, y_pred, average="macro", labels=np.unique(y_pred)), 
		recall_score(y_test, y_pred, average="macro", labels=np.unique(y_pred)))


def evaluate():
	MODELS_PATH = "data/models"

	path_dt = Path(f"{MODELS_PATH}/DT.model")
	path_word2vec = Path(f"{MODELS_PATH}/word2vec.model")
	path_mapping = Path(f"{MODELS_PATH}/libs_mapping.model")

	if path_dt.exists() and path_word2vec.exists() and path_mapping.exists():
		print("Loading the trained data ...")
		dt = pickle.load(open(f"{MODELS_PATH}/DT.model", "rb"))
		model_w2vec = pickle.load(open(f"{MODELS_PATH}/word2vec.model", "rb"))
		libs_mapping = pickle.load(open(f"{MODELS_PATH}/libs_mapping.model", "rb"))

		models_test = [
			("DT", dt),
		]
		print("Done !")

		qualitas_data = list()
		print("Reading data file ...")
		with open("data/qualitas.txt") as f:
			while True:
				line = f.readline()
				if not line:
					break
				else:
					line = line.strip()
					line = line.split()

					if len(line) > 2:
						qualitas_data.append(line)

		print("Done !")

		print("Calculating Precision and Recall for the data ...")
		tops = [1, 3, 5]

		for name, selected_model in models_test:
			print(name)
			for top in tops:
				precision_k, recall_k = benchmark(qualitas_data, selected_model, model_w2vec, libs_mapping, top)
				print(f"Precision@{top}={precision_k} Recall@{top}={recall_k}")
			print()
		print("Done !")
	else:
		print()
		print("ERROR: One or several trained models are required to do this step!")
		print("Please, complete previous steps to get the required models")
		print()
