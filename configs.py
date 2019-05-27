import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument("--ID", type=str, default="none", help="ID referring to an experiment")
    parser.add_argument("--dataset", type=str, default="amazon", help="amazon|microsoft|macys|nordstrom|collection \
        use _ to connect more than one of the four companies for combinations")
    parser.add_argument("--data_source", type=str, default="glassdoor", choices={"glassdoor"})
    parser.add_argument("--mode", type=str, default="train", choices={"train", "val", "test"}, help="train|val|test")
    parser.add_argument("--classifier", type=str, default="svm", choices={"svm","nbayes", "cnn", "bert"}, help="which classifier to use")
    # learning
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--init_iter", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=200000)
    parser.add_argument("--save_iter", type=int, default=5000)
    parser.add_argument("--snapshot_iter", type=int, default=1)

    # hyper parameters
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--word_embedding", type=str, default="GloVe")
    parser.add_argument("--rating_mode", type=int, choices = {2, 3, 5}, default=3, help="binary tenary or 5-star rating")
    parser.add_argument("--rebalance", type=bool, default=False)
    args = parser.parse_args()
    return args


system_configs = parse_args()