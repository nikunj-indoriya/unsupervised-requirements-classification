import pandas as pd

df = pd.read_csv("results/promise_supervised_full.csv")

# normalize naming
df["clustering"] = "supervised"

# remove duplicates
df = df.drop_duplicates(subset=["k", "class_subset"])

df.to_csv("results/promise_supervised_clean.csv", index=False)