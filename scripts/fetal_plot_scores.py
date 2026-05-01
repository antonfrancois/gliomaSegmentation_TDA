# %% Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Open file

filename = (
    "results/fetal_sat_TDAseg_scores_[0.5, 0, 0.3, 0.1, 0.5, 0.02, 'width']_len18.csv"
)
df = pd.read_csv(filename)

# Print mean score
segmentation_labels = ["CP"]
for label in segmentation_labels:
    seg = [df.iloc[j][label] for j in range(len(df))]
    for nb_decimals in [2, 3]:
        mean_score = round(np.mean(seg), nb_decimals)
        std_score = round(np.std(seg), nb_decimals)
        median_score = round(np.median(seg), nb_decimals)
        msg_scores = f"${mean_score}\\pm{std_score}$, median {median_score}"
        msg = label + " - TDA: " + msg_scores
        print(msg)

    # Print min, argmin, max, argmax
    print(f"Min: {min(seg)}, argmin: {21+np.argmin(seg)}")
    print(f"Max: {max(seg)}, argmax: {21+np.argmax(seg)}")


# %%  Plot Dice curve

eps = 0.03
plt.figure(figsize=(6, 2))
plt.plot(range(21, 38 + 1), df["CP"], "black")
plt.scatter(range(21, 38 + 1), df["CP"], c="black")
plt.xlim(21 - 0.5, 38 + 1)
plt.xticks(list(range(21, 38 + 1, 2)) + [38])
plt.ylim(min(df["CP"]) - eps, max(df["CP"]) + eps)
plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8])
plt.xlabel("Gestational week", fontsize=12)
plt.ylabel("Dice", fontsize=12)
plt.tight_layout()
plt.savefig("results/fetal_scores.pdf", bbox_inches="tight")
plt.show()
