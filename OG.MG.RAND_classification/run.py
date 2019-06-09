from model import CNN
import utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
#from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy
import sys
sys.path.append('../sentiment_analysis/')
import source.utils
import os

"""
data to be saved before run:

/data/glove.6B
/data/GLASSDOOR/pos_rating_pros_and_cons
/data/GLASSDOOR/neg_rating_pros_and_cons
"""

GLOVE_HOME = os.path.join('data', 'glove.6B')
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def train(data, params):
    if params["MODEL"] != "rand":
        # load word2vec
        #print("loading word2vec...")
        #word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        print("loading glove...")
        word_vectors = source.utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.50d.txt'))

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.keys():
                wv_matrix.append(word_vectors[word])
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 50).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 50).astype("float32"))
        wv_matrix.append(np.zeros(50).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    #model = CNN(**params).cuda(params["GPU"])
    model = CNN(**params).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_f1 = 0
    max_dev_f1 = 0
    max_test_f1 = 0
    print("start training...")
    for e in range(params["EPOCH"]):
        print("begin epoch {}.".format(e + 1))
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]

            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

            #batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            #batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])
            batch_x = Variable(torch.LongTensor(batch_x)).to(device)
            batch_y = Variable(torch.LongTensor(batch_y)).to(device)

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        dev_f1 = test(data, model, params, mode="dev")
        #test_f1 = test(data, model, params)
        print("epoch:", e + 1, "/ dev_f1:", dev_f1)#, "/ test_f1:", test_f1)

        if params["EARLY_STOPPING"] and dev_f1 <= pre_dev_f1 - 0.01 and e >= 10:
            print("early stopping by dev_f1!")
            break
        else:
            pre_dev_f1 = dev_f1

        if dev_f1 > max_dev_f1:
            max_dev_f1 = dev_f1
            #max_test_f1 = test_f1
            best_model = copy.deepcopy(model)

    print("max dev f1:", max_dev_f1) #, "test f1:", max_test_f1)
    return best_model


def test(data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    #x = Variable(torch.LongTensor(x)).cuda(params["GPU"])

    #x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    x = Variable(torch.LongTensor(x)).to(device)
    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    # print(x[:10])
    # print(pred[:10])
    # print(y[:10])
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)
    #print("accuracy:", acc)
    f1 = f1_score(y, pred, average='macro')  
    print(classification_report(y, pred))
    print("{} accuracy: {}".format(mode, acc))
    return f1#acc


def main():
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="static", help="available models: rand, static (fixed glove), non-static(updating glove)")
    parser.add_argument("--dataset", default="GLASSDOOR", help="available datasets: GLASSDOOR, MR, TREC")
    parser.add_argument("--save_model", default=True, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=True, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.05, type=float, help="learning rate")
    parser.add_argument("--gpu", default=1, type=int, help="the number of gpu to be used")

    options = parser.parse_args()
    data = getattr(utils, f"read_{options.dataset}")()

    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 16,
        "WORD_DIM": 50,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        #model = utils.load_model(params).cuda(params["GPU"])
        model = utils.load_model(params).to(device)

        test_f1 = test(data, model, params)
        print("test f1:", test_f1)


if __name__ == "__main__":
    main()
