import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from sklearn.pipeline import Pipeline
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataset = pd.read_csv("../dataset/toxic_comments.csv", encoding="ISO-8859-1")
# Dependent variables
labels = list(set(dataset.columns) - {'id', 'comment_text'})

print("\n---Number of comments in each category:")
comment_counts_by_category_dic = {"category": [], "comments_count": []}

comment_counts_by_category = dataset[labels].sum()

comment_counts_by_category.plot(x="category", y="comments_count", kind="bar", legend=False, grid=True, figsize=(8, 5))
plt.title("\n---Number of comments per category")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)
plt.show()
print(comment_counts_by_category)

print("\n---How many comments have multi labels?")
comments_by_label_counts = dataset.iloc[:, 2:].sum(axis=1)
print(comments_by_label_counts)
x = comments_by_label_counts.value_counts()
# plot
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=x.index, y=x.values)
plt.title("Multiple categories per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)
plt.show()

print('\n---Percentage of comments that are not labelled:')
print(len(dataset[
              (dataset['toxic'] == 0)
              & (dataset['severe_toxic'] == 0)
              & (dataset['obscene'] == 0)
              & (dataset['threat'] == 0)
              & (dataset['insult'] == 0)
              & (dataset['identity_hate'] == 0)
          ]) / len(dataset)
      )


print("\n---Split train dataset into train and validation dataset:")

train_set, test_set = train_test_split(dataset, random_state=42, test_size=0.33, shuffle=True)

print("\n---Use pipeline to build model:")
# The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.
# Sequentially apply a list of transforms and a final estimator.
# Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods.
# The final estimator only needs to implement fit.

# Define a pipeline combining a text feature extractor with multi lable classifier
# OneVsRest strategy can be used for multi-label learning,
# where a classifier is used to predict multiple labels for instance.
# Naive Bayes supports multi-class, but we are in a multi-label scenario,
# therefore, we wrap Naive Bayes in the OneVsRestClassifier.
naive_bayes_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))),
])


for label in labels:
    print('... Processing {}'.format(label))
    # train the model using X_dtm & y
    naive_bayes_pipeline.fit(train_set["comment_text"], train_set[label])
    # compute the testing accuracy
    prediction = naive_bayes_pipeline.predict(test_set["comment_text"])
    print(prediction)
    print('Test accuracy is {}'.format(accuracy_score(test_set[label], prediction)))


