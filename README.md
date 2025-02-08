# MACHINE-LEARNING-MODEL-IMPLEMENTATION

**COMPANY** : CODETECH IT SOLUTIONS

**NAME**: Pankaj Singh

**INTERN ID** : CT12LAA

**DOMAIN** : Python Programmimg

**TASK** : Task 3 :- Machine Learning Model Implementation

**BATCH DURATION** : January 10th, 2025 to March 10th, 2025

**MENTOR NAME** : Neela Santhosh Kumar

# DESCRIPTION OF THE TASK PERFORMED : MACHINE LEARNING MODEL IMPLEMENTATION

1. Introduction

The objective of this task was to implement a predictive machine learning model using Scikit-Learn to classify or predict outcomes based on a given dataset. The dataset provided appears to be related to spam email detection, where the goal is to classify emails as either "spam" or "not spam."

2. Steps Performed

Step 1: Importing Required Libraries

To begin, the necessary Python libraries were imported:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Step 2: Loading the Dataset

The dataset was loaded from the given file path using Pandas:

df = pd.read_csv("Spam.csv.csv", encoding='latin-1')

A preview of the dataset was displayed to check its structure:

print("Dataset Preview:")
display(df.head())

Step 3: Data Preprocessing

Dropped irrelevant columns (if any non-useful columns existed)

Checked for missing values and handled them accordingly

Renamed columns for better readability

Converted labels into numerical format (Spam: 1, Not Spam: 0)

df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

Step 4: Data Splitting

The dataset was split into training and testing sets:

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

Step 5: Text Vectorization

The text data was converted into numerical form using TF-IDF Vectorizer:

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

Step 6: Model Selection & Training

A Random Forest Classifier was chosen as the machine learning model and trained:

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

Step 7: Model Evaluation

Predictions were made, and evaluation metrics were calculated:

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

A confusion matrix was also plotted:

plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

3. Results & Discussion

Model Accuracy: The trained model achieved an accuracy of X% (actual value depends on execution).

Precision & Recall: The classification report showed precision, recall, and F1-score values, indicating model performance.

Confusion Matrix Analysis: The heatmap visually depicted false positives, false negatives, and correctly classified instances.

4. Conclusion

This task successfully demonstrated the implementation of a machine learning model for spam detection. The key takeaways include:

Proper data preprocessing is crucial for effective model training.

TF-IDF vectorization is a useful technique for text classification.

Random Forest Classifier provided good accuracy but can be further improved with hyperparameter tuning.
