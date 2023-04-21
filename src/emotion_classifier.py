#### IMPORT PACKAGES
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from transformers import pipeline
import seaborn as sns

# add parent directory to the Python path
sys.path.append("..")

#### INITIALIZE HUGGINGFACE PIPELINE
classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores=True)

##### READ IN DATA
filename = os.path.join("..","in", "fake_or_real_news.csv")
data = pd.read_csv(filename, index_col = 0)
data = data.reset_index(drop = True) # to prevent errors in the for loop

headlines = data["title"]
fake_or_real = data["label"] # label here refers to fake/real

#### PROCESS DATA

# create empty df to store results
results_df = pd.DataFrame(columns=["real/fake", "emotion", "count", "proportion"])

# create empty df to store results
results_df = pd.DataFrame(columns=["real/fake", "emotion", "count", "proportion"])

# loop that processes each headline
for i, headline in enumerate(headlines):
    preds = classifier(headline)
    top_emotion = max(preds[0], key=lambda x:x['score'])
    predicted_emotion = top_emotion['label']
    news_label = fake_or_real[i]

    # append predicted emotion and real/fake label to the df
    results_df.loc[len(results_df)] = [news_label, predicted_emotion, 1, None]

# group results by "real/fake" and "emotion" and sum the counts
grouped_results_df = results_df.groupby(["real/fake", "emotion"], as_index=False)["count"].sum()

# calculate the total count for each "real/fake" group
total_count = grouped_results_df.groupby("real/fake", as_index=False)["count"].sum()

# calculate the proportion for each group
results_df["proportion"] = results_df.apply(lambda x: x["count"] / total_count[total_count["real/fake"] == x["real/fake"]]["count"].values[0], axis=1)

# group the results by "real/fake" and "emotion" again and sum the counts and proportions
grouped_results_df = results_df.groupby(["real/fake", "emotion"], as_index=False).agg({"count": "sum", "proportion": "sum"})

# Round to 3 decimal places
grouped_results_df = grouped_results_df.round(3)

# create "out" folder (if it doesn't exist)
out_dir = os.path.join("..", "out")
os.makedirs(out_dir, exist_ok=True)

# save results to a CSV file
grouped_results_df.to_csv(os.path.join(out_dir, "emotion proportions.csv"), index=False)

#### VISUALIZE

# create pivot table to arrange the data in the desired format
pivot_df = grouped_results_df.pivot(index="emotion", columns="real/fake", values="proportion")
# sort pivot table by index
pivot_df = pivot_df.reindex(index=sorted(pivot_df.index))

# create bar chart
ax = pivot_df.plot(kind='bar', width=0.8)
ax.set_xlabel("Emotion")
ax.set_ylabel("Proportion")
ax.set_title("Proportions of Emotions for Real vs Fake Headlines")

# add the legend
plt.legend(title="Headline type", loc='upper right', bbox_to_anchor=(1.3, 1))

# show the plot
plt.show()

# create "out" folder (if it doesn't exist)
out_dir = os.path.join("..", "out")
os.makedirs(out_dir, exist_ok=True)

# save plot in the "out" folder
ax.figure.savefig(os.path.join(out_dir, "emotion proportions.png"), bbox_inches='tight')