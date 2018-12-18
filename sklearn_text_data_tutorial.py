from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import os
from sklearn.feature_extraction.text import CountVectorizer


#We will work with 4 categories
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']


##############################   Loading data    ##############################

print("\n ---- Loading 20 newsgroups dataset filtering categories: %r" % categories)

data_train = fetch_20newsgroups(data_home=os.path.dirname(os.path.realpath(__file__))+'/data/twenty_newsgroups/', subset='train', categories=categories, shuffle=True, random_state=42)
data_test = fetch_20newsgroups(data_home=os.path.dirname(os.path.realpath(__file__))+'/data/twenty_newsgroups/', subset='test', categories=categories,shuffle=True, random_state=42)


print('data loaded, %d documents for training' % len(data_train.data))

print("\n Example of one document in the training set:")
print("\n ----------------------------------- \n")
print(data_train.data[0])
print("\n ----------------------------------- \n")
print("Its classification: "+data_train.target_names[data_train.target[0]]) 


#################################   Extracting Features    ############################

#Bag of words: assign a number to each word in the corpus and store in x[i,j] the ocurrences of word j in document i

print("\n ---- Extracting features from the training data")

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data_train.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'algorithm'))

#################################   From occurrences to frequencies   ############################

# longer documents will have higher count values
# so lets  divide the number of occurrences of each word in a document by the total number of words in the document
# also, lets downscale weights for words that occur in many documents and are therefore less informative 
# tf= term frequency, and tf-idf = term frecuency times inverse document frequency

print("\n---- Normalizing features to a tf-idf representation")

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


#################################   Training   ############################

# we fit the classifier with the frequencies of each word in a document and the classification of the document

print("\n---- Training classifier")

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, data_train.target)




#################################   Predict   ############################

#To predict, we extract the same features from the test documents
#Or, we use a pipeline with the original documents


print("\n---- Predicting 2 new documents")

new_docs = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(new_docs)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(new_docs, predicted):
    print('%r => %s' % (doc, data_train.target_names[category]))



from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(data_train.data, data_train.target)



#or directly use the pipeline with the original test documents
print("\n---- Predicting with the test dataset")
docs_test = data_test.data
predicted = text_clf.predict(docs_test)


#################################   Statistics   ############################

#Calculating the precision
print("\n Precision Naive Bayes classifier:" )  
print(np.mean(predicted == data_test.target))


print("\n Classification Report")
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(data_train.data, data_train.target)  

predicted = text_clf.predict(docs_test)
print(np.mean(predicted == data_test.target)   )




print("\n Confusion matrix")
from sklearn import metrics
print(metrics.classification_report(data_test.target, predicted,
    target_names=data_test.target_names))
print(metrics.confusion_matrix(data_test.target, predicted))


#################################   Parameter tunning   ############################

#we run an exhaustive search of the best parameters on a grid of possible values

from sklearn.model_selection import GridSearchCV
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)

gs_clf = gs_clf.fit(data_train.data[:400], data_train.target[:400])

print(data_train.target_names[gs_clf.predict(['God is love'])[0]])

gs_clf.best_score_                                  

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

