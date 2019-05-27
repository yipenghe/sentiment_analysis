import os
import json
import argparse
import numpy as np
from utils.dataLoaders import load_glassdoor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from configs import system_configs
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


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
rebalance = system_configs.rebalance



def baseline():
    #baseline bag of words counter feature extractor, using SVM
    #need rebalancing
    rating_list = [2, 3, 5]
    with open("baseline_result","w",encoding="utf-8") as f:
        description = "This file records the result of the baseline results, \nconfigurations:\n\
        test data: The first 10000 pros prompt and 10000 cons prompt in collections.glassdoor.csv\n\
        train data: Amazon, Mircosoft, Macys and Nordstrom training data,\n \
        training data are rebalanced for all labels, training data is skewed as most ratings range from 3-5\n\
        only basic preprocessing is performed, using simple count features as baselines\n\
        "
        f.write(description + "\n")
        for rating_mode in rating_list:
            f.write("RUNNING EXPERIMENT FOR RATING LEVEL: " + str(rating_mode)+"\n")
            test_pros_X, test_pros_y = load_glassdoor("collections", "train", rating_mode, "pros", labelType = "sa")
            test_cons_X, test_cons_y = load_glassdoor("collections", "train", rating_mode, "cons", labelType = "sa")
            print(Counter(test_pros_y))
            print(Counter(test_cons_y))
            f.write("Label distribution for test data")
            f.write(str(Counter(test_pros_y)))
            f.write("\n")
            f.write(str(Counter(test_cons_y)))
            f.write("\n")
            print("============================================")
            print("using only pros data to train a model")
            f.write("=======================================\n")
            f.write("Using only pros data to train a model\n")
            X, y = load_glassdoor("all", "train", rating_mode, "pros", labelType = "sa", rebalance = True)
            #print(sum(y))
            #print(sum(test_y))
            vectorizer =CountVectorizer()
            encoded_X = vectorizer.fit_transform(X).toarray()
            test_X_pros_encoded = vectorizer.transform(test_pros_X).toarray()
            test_X_cons_encoded = vectorizer.transform(test_cons_X).toarray()
            clf = SGDClassifier(max_iter=1000, tol=1e-3)
            if rebalance:
                ros = RandomOverSampler(random_state = 42)
                encoded_X, y = ros.fit_resample(encoded_X, y)
                print(Counter(y))
            clf.fit(encoded_X, y)
            #print(classification_report(clf.predict(encoded_X), y))
            report_pros = classification_report(clf.predict(test_X_pros_encoded), test_pros_y)
            report_cons = classification_report(clf.predict(test_X_cons_encoded), test_cons_y)
            print("result:")
            print(report_pros)
            print(report_cons)
            f.write("pros test result:\n")
            f.write(report_pros+"\n")
            f.write("cons test result:\n")
            f.write(report_cons+"\n")
            print("============================================")
            print("using only cons data to train a model")
            f.write("=======================================\n")
            f.write("Using only cons data to train a model\n")
            X, y = load_glassdoor("all", "train", rating_mode, "cons", labelType = "sa", rebalance = True)
            vectorizer =CountVectorizer()
            encoded_X = vectorizer.fit_transform(X).toarray()
            test_X_pros_encoded = vectorizer.transform(test_pros_X).toarray()
            test_X_cons_encoded = vectorizer.transform(test_cons_X).toarray()
            clf = SGDClassifier(max_iter=1000, tol=1e-3)
            if rebalance:
                ros = RandomOverSampler(random_state = 42)
                encoded_X, y = ros.fit_resample(encoded_X, y)
                print(Counter(y))
            clf.fit(encoded_X, y)
            #print(classification_report(clf.predict(encoded_X), y))
            report_pros = classification_report(clf.predict(test_X_pros_encoded), test_pros_y)
            report_cons = classification_report(clf.predict(test_X_cons_encoded), test_cons_y)
            print("result:")
            print(report_pros)
            print(report_cons)
            f.write("pros test result:\n")
            f.write(report_pros+"\n")
            f.write("cons test result:\n")
            f.write(report_cons+"\n")
            print("============================================")
            print("using half pros half cons data to train a model")
            f.write("=======================================\n")
            f.write("Using half pros half cons data to train a model\n")
            X, y = load_glassdoor("Amazon_Microsoft", "train", rating_mode, "pros_cons", labelType = "sa", rebalance = True)
            vectorizer =CountVectorizer()
            encoded_X = vectorizer.fit_transform(X).toarray()
            test_X_pros_encoded = vectorizer.transform(test_pros_X).toarray()
            test_X_cons_encoded = vectorizer.transform(test_cons_X).toarray()
            clf = SGDClassifier(max_iter=1000, tol=1e-3)
            if rebalance:
                ros = RandomOverSampler(random_state = 42)
                encoded_X, y = ros.fit_resample(encoded_X, y)
            clf.fit(encoded_X, y)
            #print(classification_report(clf.predict(encoded_X), y))
            report_pros = classification_report(clf.predict(test_X_pros_encoded), test_pros_y)
            report_cons = classification_report(clf.predict(test_X_cons_encoded), test_cons_y)
            print("result:")
            print(report_pros)
            print(report_cons)
            f.write("pros test result:\n")
            f.write(report_pros+"\n")
            f.write("pros test result:\n")
            f.write(report_cons+"\n")


def main():
    #baseline()
    X, y = load_glassdoor("collections","train", 2, "pros", labelType="sa")
    print(len(X))
    with open("pros_from_collection", "w", encoding="utf-8") as f:
        for text in X:
            f.write(text+"\n")


if __name__ == "__main__":
    main()