import csv
import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import SGDClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import classification_report
def load_glassdoor_dataset(datasets, mode, rating, typeOfReview):
    """
    params:
        datasets(str): a string of form "Amazon_Microsoft",
            denotes the specific companies to include, if "collection",
            will load the collection.glassdoor.csv file
        mode(str): train|val|test data
        rating(int): binary, tenary or 5-star rating indicator, should only be 2, 3 or 5
        typeOfReview(str): "pros":only load pros, "cons": only load cons,
                    "pros_cons":load both separately, treat as two examples,
                    "concat": concate both as one sample
    returns:
        X: data list of list of strings(sentences)
        y: labels: list of floating type integers
    """
    assert(mode == "train" or mode == "val" or mode =="test")
    assert(rating in [2, 3, 5])
    assert(datasets)
    dataset_list = datasets.split("_")
    data_path = ("data/glassdoor/")

    X = []
    y = []
    for dataset in dataset_list:
        if dataset == "collections":
            #needs different loading
            continue
        csv_name = data_path +dataset+".glassdoor.csv"
        df = pd.read_csv(csv_name, usecols=["pros", "cons", "rating_overall"])
        if rating == 2:
            df.loc[df['rating_overall'] < 3.0, 'rating_overall'] = 0 #negative
            df.loc[df['rating_overall'] >= 3.0, 'rating_overall'] = 1 #positive
        elif rating == 3:
            df.loc[df['rating_overall'] < 3.0, 'rating_overall'] = 0 #negative
            df.loc[df['rating_overall'] == 3.0, 'rating_overall'] = 1 #netural
            df.loc[df['rating_overall'] > 3.0, 'rating_overall'] = 2 #positive
        labels = list(df['rating_overall'])
        if typeOfReview == "pros" or typeOfReview == "pros_cons":
            X.extend(list(df["pros"]))
            y.extend(labels)
        if typeOfReview == "cons" or typeOfReview == "pros_cons":
            X.extend(list(df["cons"]))
            y.extend(labels)

    return X, y
# X, y = load_glassdoor_dataset("Amazon", "train", 2, "pros")
# test_X, test_y = load_glassdoor_dataset("Microsoft", "train", 2, "pros")
# vectorizer =CountVectorizer()
# encoded_X = vectorizer.fit_transform(X).toarray()
# clf = SGDClassifier(max_iter=1000, tol=1e-3)
# clf.fit(encoded_X, y)
# clf2 = GaussianNB()
# clf2.fit(encoded_X, y)
# test_X_encoded = vectorizer.transform(test_X).toarray()
# print(classification_report(clf.predict(test_X_encoded), test_y))
# print(classification_report(clf2.predict(test_X_encoded), test_y))