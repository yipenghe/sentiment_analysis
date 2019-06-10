
# Convolutional Neural Networks for Sentence Classification

This is the implementation of [Convolutional Neural Networks for Sentence Classification (Y.Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181) on **Pytorch**.

We modified it to train and test on Glassdoor sentiment prediction task with our SG embeddings.
 
## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.2.
- GPU: GTX 1080


## Requirements
Also you should follow library requirements specified in the **requirements.txt**.

    numpy==1.12.1
    gensim==2.3.0
    scikit_learn==0.19.0

## Execution
Please note that we don't upload the Glassdoor data set. If you would like to run the experiments, please ask me for the data.


> python run.py --help

	usage: run.py [-h] [--mode MODE] [--model MODEL] [--dataset DATASET]
				  [--save_model] [--early_stopping] [--epoch EPOCH]
				  [--learning_rate LEARNING_RATE] [--gpu GPU]

	-----[CNN-classifier]-----

	optional arguments:
	  -h, --help            show this help message and exit
	  --mode MODE           train, test
	  --model MODEL         available models: rand, static (fixed glove), non-static(updating glove), pros_cons (pros/cons embeddings)
	  --dataset DATASET     available datasets: GLASSDOOR, MR, TREC
	  --save_model          whether saving model or not
	  --early_stopping      whether to apply early stopping by checking f1
	  --epoch EPOCH         number of max epoch
	  --learning_rate LEARNING_RATE
							learning rate
	  --gpu GPU             the number of gpu to be used

 
 
## Quick Start:

>train: python run.py --model pros_cons --dataset GLASSDOOR --gpu 0 --mode train

>test: python run.py --model pros_cons --dataset GLASSDOOR --gpu 0 --mode test
