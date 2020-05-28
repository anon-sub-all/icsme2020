require("PMCMR")
require("tsutils")

mcc_results <- read.csv("../data/mcc_values.csv",
                        header = TRUE,
                        sep = ",",
                        dec = ".")

mcc_matrix <- as.matrix(mcc_results)
friedman.test(mcc_matrix)

# Results show highly significant null hypothesis rejection
nemenyi(mcc_matrix, conf.level=0.99, plottype = "mcb")
