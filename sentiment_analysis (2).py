# Movie Review Sentiment Analysis
# Educational Practice 2025-2026
# Team project

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


# dataset - positive and negative movie reviews
# we wrote these manually to have more control over the data

positive_reviews = [
    "I really enjoyed this movie, the story was interesting and kept me watching till the end",
    "Great film overall, the actors did a good job and the plot made sense",
    "One of the better movies I watched this year, would definitely recommend it",
    "The story was touching and I liked how everything connected at the end",
    "Pretty good movie, nothing too crazy but entertaining from start to finish",
    "I liked the characters a lot, felt like real people going through real problems",
    "Solid film, good pacing and the ending was satisfying",
    "The acting was really convincing, especially the main character",
    "Enjoyed every minute of it, the script was well written",
    "A feel good movie that actually delivers, left the cinema happy",
    "Surprisingly good, did not expect to like it as much as I did",
    "The cinematography was beautiful and the music fit perfectly",
    "Watched it twice already, gets better the second time",
    "Really well made, the director knew what kind of story he wanted to tell",
    "Fun and engaging, good choice for movie night",
    "The tension built up nicely and the climax was worth waiting for",
    "Strong performances from the whole cast, not just the leads",
    "I was emotionally invested by the second act, rare for this kind of film",
    "Clean storytelling without unnecessary subplots, refreshing",
    "Definitely going on my rewatch list, loved the atmosphere",
    "The dialogue felt natural, not forced or overly dramatic",
    "Brilliant from beginning to end, one of those films you remember",
    "The plot had some unexpected turns that actually made sense",
    "Heartfelt and genuine, the kind of movie that stays with you",
    "Very well paced, never felt bored or like a scene dragged on",
    "The chemistry between the leads was believable and fun to watch",
    "Good movie with a meaningful message that does not feel preachy",
    "Technical aspects were excellent, sound design especially stood out",
    "Would recommend to anyone who likes thoughtful character driven stories",
    "Watched with my family and everyone enjoyed it which is rare",
    "The script had real wit and humor without trying too hard",
    "Emotional but not manipulative, which I appreciate",
    "The world building was done subtly and effectively",
    "One of the most enjoyable films I have seen in a while",
    "The ending surprised me in the best way possible",
    "Great cast, great direction, great story, what more do you need",
    "It made me think about things after leaving the theater",
    "Genuinely funny in parts without losing its dramatic weight",
    "Loved the visual style and the color grading throughout",
    "A rare film that actually lives up to the hype",
    "The pacing was perfect, never rushed and never slow",
    "Really liked how complex the main character was written",
    "An honest and well crafted story about real human experience",
    "The score was incredible, enhanced every emotional scene",
    "Gripping from the first scene, hard to look away",
    "One of those movies where every element comes together perfectly",
    "Thoughtful and entertaining in equal measure",
    "The direction was confident and the vision was clear",
    "Left me with a smile and a lot to think about",
    "Absolutely loved it, already recommended it to three people",
]

negative_reviews = [
    "I was really bored watching this, nothing interesting happened for the first hour",
    "The plot made no sense and the ending was confusing and unsatisfying",
    "Wasted two hours of my life, would not recommend to anyone",
    "The acting was stiff and the dialogue felt fake and unnatural",
    "Disappointing film, expected much more based on the trailer",
    "Could not finish it, turned it off after 40 minutes",
    "The story had potential but was executed very poorly",
    "Characters were flat and I did not care what happened to them",
    "Too slow and too long, they could have cut 45 minutes easily",
    "The CGI looked cheap and took me out of the experience",
    "Predictable from start to finish with no surprises at all",
    "The script was full of cliches and lazy writing",
    "The director clearly had no idea what tone to go for",
    "Bad pacing, bad editing, and somehow still managed to be boring",
    "The lead actor seemed uncomfortable in the role the whole time",
    "Felt like a student film with a big budget, not in a good way",
    "The plot holes were impossible to ignore, just sloppy work",
    "Nothing about this movie felt genuine or earned",
    "Loud and chaotic without any real purpose or story",
    "One of the worst films I have seen in recent memory",
    "The humor fell flat every single time, painfully unfunny",
    "Weak story propped up by flashy visuals that get old fast",
    "The supporting cast was wasted in completely pointless roles",
    "Felt like they were making it up as they went along",
    "The third act completely fell apart and ruined what little goodwill it had",
    "Technically fine but emotionally completely empty",
    "I genuinely did not understand what this movie was trying to say",
    "The villain had no motivation and the hero was unlikeable",
    "Terrible soundtrack that clashed with every single scene",
    "So many subplots that went nowhere it became exhausting",
    "Felt like a cash grab sequel nobody asked for",
    "The romance subplot was cringeworthy and slowed everything down",
    "Poorly written female characters in an otherwise forgettable film",
    "The tone was all over the place, could not tell if it was serious or not",
    "The last twenty minutes felt like a different and worse movie",
    "No emotional payoff despite setting things up for two hours",
    "The exposition was handled so clumsily it was almost funny",
    "I looked at my phone at least five times, not a good sign",
    "Tired tropes with nothing new or interesting added to them",
    "The kind of movie that makes you question your taste for watching it",
    "Hollow spectacle with no soul underneath the action sequences",
    "Even fans of the genre will probably find this underwhelming",
    "The editing made some scenes genuinely hard to follow",
    "Too long, too loud, too dumb, not enough of anything good",
    "A waste of a talented cast on a script that did not deserve them",
    "I kept hoping it would get better and it never did",
    "The trailer showed all the good parts and there were not many",
    "Forgettable in every way, will not remember it in a week",
    "Somehow managed to make an interesting premise completely boring",
    "Sat through the whole thing and felt nothing, that says it all",
]

# duplicate and shuffle to get 2000 samples
all_reviews = []
all_labels = []

for i in range(20):
    for r in positive_reviews:
        all_reviews.append(r)
        all_labels.append(1)
    for r in negative_reviews:
        all_reviews.append(r)
        all_labels.append(0)

# shuffle
combined = list(zip(all_reviews, all_labels))
np.random.seed(42)
np.random.shuffle(combined)
all_reviews, all_labels = zip(*combined)
all_reviews = list(all_reviews)
all_labels = list(all_labels)

print("Dataset loaded")
print(f"Total reviews: {len(all_reviews)}")
print(f"Positive: {all_labels.count(1)}  Negative: {all_labels.count(0)}")


# text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


print("\nCleaning text...")
cleaned = [clean_text(r) for r in all_reviews]


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    cleaned, all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")


# tfidf vectorizer
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# train models
print("\nTraining models...\n")

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, C=1.0, random_state=42),
    'Naive Bayes': MultinomialNB(alpha=0.5),
    'Linear SVM': LinearSVC(C=1.0, max_iter=500, random_state=42),
}

results = {}
trained = {}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc  = round(accuracy_score(y_test, preds) * 100, 1)
    prec = round(precision_score(y_test, preds) * 100, 1)
    rec  = round(recall_score(y_test, preds) * 100, 1)
    f1   = round(f1_score(y_test, preds) * 100, 1)

    results[name] = {
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'preds': preds
    }
    trained[name] = model

    print(f"--- {name} ---")
    print(classification_report(y_test, preds, target_names=['Negative', 'Positive']))


# pick best model by f1
best_name = max(results, key=lambda k: results[k]['f1'])
best_model = trained[best_name]
print(f"Best model: {best_name} (F1 = {results[best_name]['f1']}%)")


# save model
os.makedirs('model', exist_ok=True)
with open('model/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Model saved to model/")


# plots
os.makedirs('plots', exist_ok=True)

# 1. label distribution
counts = [all_labels.count(1), all_labels.count(0)]
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(counts, labels=['Positive', 'Negative'],
       autopct='%1.0f%%', colors=['#4CAF50', '#F44336'],
       startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Dataset Distribution')
plt.tight_layout()
plt.savefig('plots/distribution.png', dpi=120)
plt.close()

# 2. model comparison bar chart
fig, ax = plt.subplots(figsize=(9, 5))
metrics = ['accuracy', 'precision', 'recall', 'f1']
labels_m = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(metrics))
w = 0.25
colors_m = ['#2196F3', '#4CAF50', '#FF9800']

for i, (name, color) in enumerate(zip(results.keys(), colors_m)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i*w, vals, w, label=name, color=color, alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{v}%', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + w)
ax.set_xticklabels(labels_m)
ax.set_ylim(60, 105)
ax.set_ylabel('Score (%)')
ax.set_title('Model Comparison')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=120)
plt.close()

# 3. confusion matrix for best model
cm = confusion_matrix(y_test, results[best_name]['preds'])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=120)
plt.close()

print("Plots saved to plots/")


# predict function
def predict(text):
    cleaned_text = clean_text(text)
    vec = vectorizer.transform([cleaned_text])
    pred = best_model.predict(vec)[0]
    return 'POSITIVE' if pred == 1 else 'NEGATIVE'


# test predictions
print("\nExample predictions:")
test_reviews = [
    "I really loved this movie, great story and acting",
    "This was terrible, boring and made no sense",
    "Pretty good film, enjoyed watching it",
    "Awful movie, waste of time completely",
    "The cinematography was stunning and the plot was engaging",
]

for review in test_reviews:
    result = predict(review)
    print(f"  [{result}] {review}")


# final summary
print("\n=== Results Summary ===")
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-" * 46)
for name, res in results.items():
    print(f"{name:<22} {res['accuracy']:>5}% {res['precision']:>5}% {res['recall']:>5}% {res['f1']:>5}%")
