import pandas as pd
from scipy.interpolate._rbfinterp_pythran import linear
from sklearn.feature_extraction.text import TfidfVectorizer
# Make sure to pip install sklearn
###
#
# Our goal is to find which machine learning model is best suited
# to predict sentiment (output) given a movie review (input).
#
#
#
#
###

# Preparing the data set
df_review = pd.read_csv('IMDB Dataset.csv')

df_positive = df_review[df_review['sentiment'] == 'positive'][:9000]
df_negative = df_review[df_review['sentiment'] == 'negative'][:1000]
df_review_imb = pd.concat([df_positive, df_negative])

# under sample the positive reviews
length_negative = len(df_review_imb[df_review_imb['sentiment'] == 'negative'])
df_review_positive = df_review_imb[df_review_imb['sentiment'] == 'positive'].sample(n=length_negative)
df_review_non_positive = df_review_imb[~(df_review_imb['sentiment'] == 'positive')]

df_review_bal = pd.concat([df_review_positive, df_review_non_positive])
df_review_bal.reset_index(drop=True, inplace=True)
df_review_bal['sentiment'].value_counts()  # should have 1000 positive and negative reviews

# Create some train and test data sets
# train set
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)

# set the independent and dependent variables
train_x = train['review']
train_y = train['sentiment']

# because test_size = 0.33, 33% of the test is used to make predictions
test_x = test['review']
test_y = test['sentiment']

# We need to turn our text data into numerical vectors
tfidf = TfidfVectorizer(stop_words='english')  # removed english stop words
train_x_vector = tfidf.fit_transform(train_x)  # fit and transform the text reviews

# Term Frequency-Inverse Document Frequency
# The TF-IDF value is computed by increasing proportionally to the number of
# time a word appears in the document and is offset by the number of documents
# in the corpus that contain the word.
pd.DataFrame.sparse.from_spmatrix(train_x_vector, index=train_x.index,
                                  columns=tfidf.get_feature_names())

test_x_vector = tfidf.transform(test_x)

# Model Selection
# This script uses a supervised, classification learning model because we have
# discrete values being predicted and we have labeled input and output data
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

print(svc.predict(tfidf.transform(['A good movie'])))  # positive
print(svc.predict(tfidf.transform(['An excellent movie'])))  # positive
print(svc.predict(tfidf.transform(['I did not like this movie at all'])))  # negative

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

###
###
###

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

# Model Evaluation

# Mean Accuracy
# svc.score(Test samples, true labels)
svc_score = svc.score(test_x_vector, test_y)
decision_score = dec_tree.score(test_x_vector, test_y)
Bayes_score = gnb.score(test_x_vector.toarray(), test_y)
logreg_score = log_reg.score(test_x_vector, test_y)
print("SVM:", svc_score)
print("Decision tree:", decision_score)
print("Naive Bayes: ", Bayes_score)
print("Logistic Regression: ", logreg_score)

# F1 score
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)
# where Recall is how many true positives were found (how complete)
# &
# where Precision is how many of the returned hits were true positive (how useful)

from sklearn.metrics import f1_score

f1 = f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None)
print("f1: positive, negative")
print("f1 scores: ", f1)

# Classification Report
from sklearn.metrics import classification_report

print(classification_report(test_y, svc.predict(test_x_vector), labels=['positive', 'negative']))

# Confusion Matrix
# the matrix goes (TP, FP, FN, TN) in a 2x2 matrix
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])

print(conf_mat)
