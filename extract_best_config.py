import pandas as pd

table = pd.read_csv("./wandb/standard_mf_table.csv")
table.drop(table.columns[[0, 1, 2, 3, 4, 11]], axis=1, inplace=True)
groups = table.groupby(by=["biased", "k", "lr", "tr_batch_size", "wd"])

for group_idx, group in enumerate(groups):
    print(group_idx)
    print(group)
    print()

# print(table.head())