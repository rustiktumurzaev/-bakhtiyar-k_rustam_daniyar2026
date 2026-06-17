# Movie Review Sentiment Analysis

This is our ML project for Educational Practice 2025-2026.

The idea is simple - given a movie review text, the model predicts whether it's positive or negative.

## How to run

Install the required libraries:

```
pip install scikit-learn pandas numpy matplotlib seaborn
```

Then just run:

```
python sentiment_analysis.py
```

It will train the models, print the results, and save the plots to the `plots/` folder.

## What the project does

1. Loads a dataset of 2000 movie reviews (1000 positive, 1000 negative)
2. Cleans the text (lowercase, remove punctuation etc)
3. Converts text to numbers using TF-IDF
4. Trains 3 models: Logistic Regression, Naive Bayes, Linear SVM
5. Compares them using accuracy, precision, recall and F1 score
6. Saves the best model so it can be used later

## Project structure

```
sentiment_analysis.py   - main code
requirements.txt        - libraries needed
README.md               - this file
plots/                  - charts generated after running
model/                  - saved model files
```

## Results

Best model was Logistic Regression. All three models performed well on the test set.

## Team

| Name | What they did |
|------|--------------|
| Member 1 | Data preparation, text preprocessing, evaluation, report |
| Member 2 | Model training, visualization, GitHub, demo video |
