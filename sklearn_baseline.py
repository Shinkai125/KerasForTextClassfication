"""
@file: sklearn_method.py
@time: 2020-12-09 17:38:38
"""

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname='SimHei.ttf')  # 设置字体

train_data = pd.read_csv('chnsenticorp/train.tsv', sep='\t')

tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
features = tfidf.fit_transform(train_data.text_a)
labels = train_data.label
print(features.shape)

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0, solver='liblinear'),
]
CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in tqdm(models):
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='f1', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

results = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1'])

sns.boxplot(x='model_name', y='f1', data=results)
sns.stripplot(x='model_name', y='f1', data=results,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

print(results.groupby('model_name').f1.mean())
