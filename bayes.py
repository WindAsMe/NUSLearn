import numpy as np
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == '__main__':
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    # print twenty_train.data
    for article in twenty_train.data[:3]:
        myindex = twenty_train.data.index(article)
    # print '\n ** * Article  # {} Label: {} ***\n\n'.format(myindex, twenty_train.target_names[twenty_train.target[myindex]])
    # print article
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    # print X_train_counts.shape
    # print X_train_counts[0]
    A = X_train_counts[0]
    # print A.data
    # print A.indptr
    # print A.indices[A.indptr[0]:A.indptr[1]]
    # for index in A.indices[A.indptr[0]:A.indptr[1]]:
    #     print count_vect.get_feature_names()[index]
    # print A.indices[A.indptr[0]: A.indptr[1]]
    # print X_train_counts
    # for index in A.indices[A.indptr[0]:A.indptr[1]]:
    #   print count_vect.get_feature_names()[index]

    # Before for statics get and dispose
# ==========================================================
    # Bayes to classify
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB().fit(X_train_counts, twenty_train.target)
    twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
    print len(twenty_test.data)
    X_test_counts = count_vect.transform(twenty_test.data)
    predicted = clf.predict(X_test_counts)
    print 'Accuracy: {}'.format(np.mean(predicted == twenty_test.target))

    # Improvement the accuracy
    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    tfidf_clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    predicted = tfidf_clf.predict(X_test_tfidf)
    print 'Accuracy(tfidf) is {}'.format(np.mean(predicted == twenty_test.target))

    # Improvement continuity
    # sw_count_vect = CountVectorizer(stop_words='english')
    # X_train_counts = sw_count_vect.fit_transform(twenty_train.data)
    # X_test_counts = sw_count_vect.transform(twenty_test.data)
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    # tfidf_clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
    # predicted = tfidf_clf.predict(X_test_tfidf)

    # Import SVM to improvement
    from sklearn.linear_model import SGDClassifier

    svm_clf = SGDClassifier(loss='hinge', penalty ='l2', alpha = 1e-3, max_iter = 5, random_state = 42)
    svm_clf.fit(X_train_tfidf, twenty_train.target)
    predicted = svm_clf.predict(X_test_tfidf)
    print 'Accuracy(SVM): {}.'.format(np.mean(predicted == twenty_test.target))