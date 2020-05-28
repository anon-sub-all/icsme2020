rm(list = ls())

library(reshape)
library(ggplot2)

data <- read.csv("../data/mcc_values.csv",
           header = TRUE,
           sep = ",",
           dec = ".")

data <- melt(data)

ggplot(data, aes(x = variable, y = value)) +
  geom_boxplot(fill="#7CAE00") +
  stat_summary(
    fun = "mean",
    geom = "point",
    shape = 8,
    size = 2,
    position = position_dodge(width = 0.75),
    color = "white"
  ) +
  xlab("") +
  ylab("MCC") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none", text = element_text(size=20))
