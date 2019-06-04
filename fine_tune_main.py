import tensorflow
from glove_fine_tuner import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument("--ID", type=str, default="exam")
    parser.add_argument("--type", type=str, default="pro")
    parser.add_argument("--doc_path", type=str, default="pros_from_collection", help="path for review data")
    parser.add_argument("--glove_path", type=str, default="data/glove.6B/glove.6B.50d.txt", help="path for glove data")
    parser.add_argument("--iter", type=int, default=1000, help="training iteration")
    parser.add_argument("--dim", type=int, default=50, help="glove dimension")
    parser.add_argument("--restrict", type=int, default=50, help="glove dimension")
    args = parser.parse_args()
    return args

system_configs = parse_args()
def main():
    ID = system_configs.ID
    train_type=system_configs.type
    doc_path = system_configs.doc_path
    glove_path=system_configs.glove_path
    iteration=system_configs.iter
    dim=system_configs.dim
    restrict=system_configs.restrict
    print(ID, train_type, doc_path, glove_path, iteration, dim)
    fine_tune_glove(ID = ID, train_type=train_type, doc_name = doc_path, glove_file = glove_path, iteration = iteration, glove_dim = dim, restrict=restrict)
    print("finished")

main()