from scripts import train_evaluate_resico_coster, get_stats_coster_data, get_mcc, train_evaluate_resico_all, evaluate_qualitas

if __name__ == '__main__':
	print("--------------------------")
	print("RESICO replication package")
	print("--------------------------")

	print("Research Question 1: ")
	print("Getting metrics for MCC of the classifiers ...")
	get_mcc.get_metrics()
	print("Done!")

	print("Research Question 2: ")
	print("Evaluating the performance of RESICO on COSTER dataset ...")
	train_evaluate_resico_coster.train_evaluate_coster("data/coster.txt")
	print("Done with the evaluation on COSTER dataset !")

	print("Getting a summary of the COSTER extended dataset ...")
	get_stats_coster_data.get_stats("data/coster_extended.txt")

	print("Evaluating the performance of RESICO on COSTER extended dataset ...")
	train_evaluate_resico_coster.train_evaluate_coster("data/coster_extended.txt")
	print("Done with the evaluation on COSTER extended dataset !")

	print("Research Question 3: ")
	print("Evaluating the performance of RESICO on Qualitas corpus of data ...")
	evaluate_qualitas.evaluate()
	print("Done with the evaluation on Qualitas dataset !")

	print("--------------------------")
