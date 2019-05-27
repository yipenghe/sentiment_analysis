import csv
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import re

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  from https://github.com/harvardnlp/sent-conv-torch/blob/master/preprocess.py
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()


def load_glassdoor(datasets="all", mode="train", rating=5, typeOfReview="pros_cons", labelType="sa",pros_cons_split=0.5, restrict= 0):
    """
    params:
        datasets(str): a string of form "Amazon_Microsoft",
            denotes the specific companies to include, if "collection",
            will load the collection.glassdoor.csv file
            "all": will load all datasets except collection, same as "Amazon_Mircosoft_NordStrom_Macys"
        mode(str): train|val|test data
        rating(int): binary, tenary or 5-star rating indicator, should only be 2, 3 or 5
        typeOfReview(str): "pros":only load pros, "cons": only load cons,
                    "pros_cons":load both separately, treat as two examples,
                    "concat": concate both as one sample
        labelType: pros_cons: load review with label as "pros" or "cons"
                    sa: load review with label as rating
        pros_cons_split: the data split between pros:cons,
    returns:
        X: data list of list of strings(sentences)
        y: labels: list of floating type integers
    """
    assert(mode == "train" or mode == "val" or mode =="test")
    assert(rating in [2, 3, 5])
    assert(datasets)
    if datasets == "all":
        datasets = "Amazon_Microsoft_NordStrom_Macys"
    dataset_list = datasets.split("_")
    data_path = ("data/glassdoor/")

    X = []
    y = []
    #for sentiment analysis type
    if labelType == "sa":
        for dataset in dataset_list:
            if dataset == "collections":
                #needs different loading
                count = Counter()
                csv_name = data_path +dataset+".glassdoor.csv"
                df = pd.read_csv(csv_name, usecols=["prompt", "text", "overall"])
                if rating == 2:
                    df.loc[df['overall'] < 3.0, 'overall'] = 0 #negative
                    df.loc[df['overall'] >= 3.0, 'overall'] = 1 #positive
                elif rating == 3:
                    df.loc[df['overall'] < 3.0, 'overall'] = 0 #negative
                    df.loc[df['overall'] == 3.0, 'overall'] = 1 #netural
                    df.loc[df['overall'] > 3.0, 'overall'] = 2 #positive
                labels = list(df['overall'])
                list_text = list(df["text"])
                for index, prompt in enumerate(list(df['prompt'])):
                    if typeOfReview == "pros" or typeOfReview == "pros_cons":
                        if prompt == "pros":
                            X.append(clean_str(list_text[index]))
                            y.append(labels[index])
                    if typeOfReview == "cons" or typeOfReview == "pros_cons":
                        if prompt == "cons":
                            X.append(clean_str(list_text[index]))
                            y.append(labels[index])
                    #if (len(X) >= 10000):
                    #    break
            else:
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
                    for pro in list(df["pros"]):
                        X.append(clean_str(pro))
                    y.extend(labels)
                if typeOfReview == "cons" or typeOfReview == "pros_cons":
                    for con in list(df["cons"]):
                        X.append(clean_str(con))
                    y.extend(labels)
    elif labelType == "pros_cons":
        #for pros cons recognition
        for dataset in dataset_list:
            if dataset == "collections":
                #needs different loading
                csv_name = data_path +dataset+".glassdoor.csv"
                df = pd.read_csv(csv_name, usecols=["prompt", "text"])
                list_prompt = list(df["prompt"])
                list_text = list(df["text"])
                for index, prompt in enumerate(list_prompt):
                    if prompt == "pros":
                        X.append(clean_str(list_text[index]))
                        y.append(1)
                    elif prompt == "cons":
                        X.append(clean_str(list_text[index]))
                        y.append(0)
                    if restrict > 0 and len(X) >= restrict:
                    #if len(X) >= 10000:
                        break
            else:
                csv_name = data_path +dataset+".glassdoor.csv"
                df = pd.read_csv(csv_name, usecols=["pros", "cons"])
                list_pros = list(df["pros"])
                total_data = len(list_pros)
                num_of_pros = int(total_data*pros_cons_split)
                num_of_cons = int(total_data - num_of_pros)
                for example in list_pros[:num_of_pros]:
                    X.append(clean_str(example))
                    y.append(1)
                list_cons = list(df["cons"])[:num_of_cons]
                for example in list_cons[:num_of_cons]:
                    X.append(clean_str(example))
                    y.append(0)
    #if (rebalance):
     ##   ros = RandomOverSampler(random_state = 42)
       ##X, y = ros.fit_resample(X, y)
    return X, y