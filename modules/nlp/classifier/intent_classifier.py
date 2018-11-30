# coding=utf8

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from modules.nlp.share_models import utils


curr_dir_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(curr_dir_path, "models")

dialogues_file = os.path.join(models_path, "dialogues.tsv")
stackoverflow_file = os.path.join(models_path, "tagged_posts.tsv")

tfidf_vec_pkl_path = os.path.join(models_path, "tfidf_vectorizer.pkl")
intent_recg_pkl_path = os.path.join(models_path, "intent_recognizer.pkl")
tag_classifier_pkl_path = os.path.join(models_path, "tag_classifier.pkl")

intent_recognizer = None
tfidf_vectorizer = None
tag_classifier = None

class Classifier:
    def __init__(self):
        pass

    sample_size = 200000

    def tfidf_features(self, X_train, X_test, vectorizer_path):
        """Performs TF-IDF transformation and dumps the model."""

        # Train a vectorizer on X_train data.
        # Transform X_train and X_test data.

        # Pickle the trained vectorizer to 'vectorizer_path'
        # Don't forget to open the file in writing bytes mode.
        tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1, 2), token_pattern='(\S+)')
        x_train_fit = tfidf_vectorizer.fit(X_train)
        X_train = tfidf_vectorizer.transform(X_train)
        X_test = tfidf_vectorizer.transform(X_test)
        pickle.dump(tfidf_vectorizer, open(vectorizer_path, 'wb'))
        return X_train, X_test


    def reload(self):
        try:
            print("Classifier Reload Started ....")
            dialogue_df = pd.read_csv(dialogues_file, sep='\t').sample(self.sample_size, random_state=0)
            stackoverflow_df = pd.read_csv(stackoverflow_file, sep='\t').sample(self.sample_size, random_state=0)
            dialogue_df['text'] = dialogue_df['text'].apply(utils.text_prepare)
            stackoverflow_df['title'] = stackoverflow_df['title'].apply(utils.text_prepare)

            # concatenate dialogue and stackoverflow examples into one sample
            X = np.concatenate([dialogue_df['text'].values, stackoverflow_df['title'].values])
            y = ['dialogue'] * dialogue_df.shape[0] + ['stackoverflow'] * stackoverflow_df.shape[0]

            # split it into train and test in proportion 9:1, use random_state=0 for reproducibility
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9,
                                                                random_state=0)  ######### YOUR CODE HERE ##########
            print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

            # transform it into TF-IDF features
            X_train_tfidf, X_test_tfidf = self.tfidf_features(X_train=X_train, X_test=X_test,
                                                         vectorizer_path=tfidf_vec_pkl_path)
            print((X_train_tfidf.shape))
            print((X_train_tfidf.ndim))

            # Train the intent recognizer using LogisticRegression on the train set with the following parameters:
            # penalty='l2', C=10, random_state=0. Print out the accuracy on the test set to check whether everything
            # looks good.

            intent_recognizer = LogisticRegression(penalty='l2', C=10, random_state=0).fit(X_train_tfidf, y_train)

            # Check test accuracy.
            y_test_pred = intent_recognizer.predict(X_test_tfidf)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            print('Test accuracy = {}'.format(test_accuracy))
            pickle.dump(intent_recognizer, open(intent_recg_pkl_path, 'wb'))

            # Programming language classification

            # We will train one more classifier for the programming-related questions. It will predict exactly one tag
            # (=programming language) and will be also based on Logistic Regression with TF-IDF features.

            # First let us prepare the data for this task.
            X = stackoverflow_df['title'].values
            y = stackoverflow_df['tag'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

            vectorizer = pickle.load(open(tfidf_vec_pkl_path, 'rb'))
            X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)
            tag_classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=5, random_state=0)).fit(X_train_tfidf,
                                                                                                            y_train)
            # Check test accuracy.
            y_test_pred = tag_classifier.predict(X_test_tfidf)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            print('Test accuracy = {}'.format(test_accuracy))

            pickle.dump(tag_classifier, open(tag_classifier_pkl_path, 'wb'))
            print("Reload Completed ...")
        except Exception as e:
            print("Exception Occurred while Reloading the Classifier ...")
            raise e

    def load(self):
        try:
            global intent_recognizer
            global tfidf_vectorizer
            global tag_classifier
            print("Loading the Classifier model files...")
            intent_recognizer = utils.unpickle_file(intent_recg_pkl_path)
            tfidf_vectorizer = utils.unpickle_file(tfidf_vec_pkl_path)
            tag_classifier = utils.unpickle_file(tag_classifier_pkl_path)
            print("Done...")
        except Exception as e:
            print("Exception Occurred while loading the classifier Models...")



if __name__ == "__main__":
    clf = Classifier()
    clf.reload()



