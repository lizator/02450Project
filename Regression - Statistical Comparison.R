df_ann = read.csv("Error_ANN.csv", header = FALSE)
df_base = read.csv("Error_baseline.csv", header = FALSE)
df_lr = read.csv("Error_linear_regression.csv", header = FALSE)

ann_vs_lr = df_ann-df_lr
ann_vs_base = df_ann-df_base
lr_vs_base = df_lr-df_base

t.test(ann_vs_lr, alternative = "two.sided", alpha=0.05)
t.test(ann_vs_base, alternative = "two.sided", alpha=0.05)
t.test(lr_vs_base, alternative = "two.sided", alpha=0.05)