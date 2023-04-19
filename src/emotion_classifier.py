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
data = data.reset_index(drop = True) # this was added to prevent errors in the for loop

headlines = data["title"]
fake_or_real = data["label"] # label here refers to fake/real


for i, headline in enumerate(headlines[:10]):
    preds = classifier(headline)
    top_emotion = max(preds[0], key=lambda x:x['score'])
    print(top_emotion['label'])
    print(fake_or_real[i])