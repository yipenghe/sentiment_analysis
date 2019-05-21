import os
import json
import argparse
import numpy as np
from utils.dataLoaders import load_glassdoor_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from configs import system_configs

def main():
    #configs of the experiment
    ID            = system_configs.ID #name of experiment
    dataset       = system_configs.dataset #dataset(s) we are using example: "amazon_microsoft" means using both amazon and microsoft data
    data_source   = system_configs.data_source #data source to run, only glassdoor for now
    batch_size    = system_configs.batch_size #batch size
    learning_rate = system_configs.learning_rate
    mode          = system_configs.mode #train val test
    classifier    = system_configs.classifier #type of classifier
    word_embedding = system_configs.word_embedding#word embedding: GloVe/Bert
    rating_mode   = system_configs.rating_mode #binary, tenary or 5 star rating

    #baseline bag of words counter feature extractor, using SVM
    #need rebalancing
    X, y = load_glassdoor_dataset("Amazon", "train", 2, "cons")
    print(sum(y))
    test_X, test_y = load_glassdoor_dataset("Microsoft", "train", 2, "cons")
    print(sum(test_y))
    vectorizer =CountVectorizer()
    encoded_X = vectorizer.fit_transform(X).toarray()
    test_X_encoded = vectorizer.transform(test_X).toarray()
    clf = SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(encoded_X, y)
    print(classification_report(clf.predict(encoded_X), y))
    print(classification_report(clf.predict(test_X_encoded), test_y))


if __name__ == "__main__":
    main()

