# Assignment 4 - Using finetuned transformers via HuggingFace

## Contribution
The code in this assignment was developed collaboratively with the help of other students and materials used in class.

## Description of assignment
In this assignment, I am using a HuggingFace pipeline to classify both real and fake news headlines based on emotion (anger, disgust, fear, joy, neutral, sadness, and surprise).

## Data
The data comes from the same *Fake or Real News* dataset that was used in Assignment 2. It is a CSV file containing newspaper articles, with one half labeled 'Fake' and the other half labeled 'True'. The data has three columns: "title," "text," and "label." You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news).

## Methods
The `emotion_classifier.py` script utilizes a HuggingFace pipeline to classify news headlines based on emotion. Here is a brief explanation of what the code does:

- Initializes the HuggingFace pipeline for text classification using the `j-hartmann/emotion-english-distilroberta-base` model.
- Reads in the data from the `fake_or_real_news.csv` file.
- Processes each headline by predicting the emotion using the classifier.
- Stores the predicted emotion and real/fake label in a dataframe.
- Groups the results by "real/fake" and "emotion" and calculates counts and proportions.
- Saves the results to a CSV file and visualizes the proportions of emotions for real vs. fake headlines in a bar chart.
- Saves the plot as an image file.

## Usage and Reproducibility
To use this code on your own device, follow these steps:

1. Clone this GitHub repository to your local device.
2. Install the required packages by navigating to the root folder and running `pip install -r requirements.txt` in your terminal.
3. Run the script by executing `python src/emotion_classifier.py` in your terminal.
4. Find the results in the "out" folder as a `.csv` file and a `.png` file.

Please note that this code was successfully executed in Coder Python 1.76.1 on uCloud. Your terminal commands may need to vary slightly depending on your device.

## Discussion of Results
The results of the classification show that the overwhelming majority of headlines were neutral for both real headlines (52%) and fake headlines (48.4%). The emotions of fake headlines closely resemble the distribution of real headlines. The findings also confirm the common understanding that news tends to focus more on negative events rather than positive ones. This is evident in the significantly lower number of headlines classified with the "joy" emotion compared to "anger," "disgust," "fear," and "sadness."
