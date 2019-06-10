# Improving Sentiment Classification with Pro/Con Structure of Reviews
This is a course project for CS224U by Di Bai and Yipeng He.

We explored the usage of Pros and Cons structure of reviews, aiming to improve the sentiment classification task. We fine tuned GloVe embedding(https://nlp.stanford.edu/projects/glove/). We release the word embeddings we fine tuned here(https://drive.google.com/drive/u/0/my-drive), please read the readMe.txt inside the folder for usage.

Below is the file_structure for data folder (not uploaded because files are large and data aren't publicly released). Unfortunately the dataset we used is also not publicly released. If you would like to use our code, please take a look at utils/dataLoader.py file, this script is used load the csv files containing reviews. You need to have header "pros", "cons", "overall_rating" in your csv file to use the code, or you can modify the code for your own needs.
```
project
│   README.md
│
│
└───data
│   │
│   │
│   │───fine_tune_docs
│   │   │
│   │   │ *_from_collection
│   │   │ *_lemma_processed_doc
│   │   │ *_stop_words
│   │
│   └───glove.6B (same as cs224U github data/glove.6B)
│   │   │ glove.6B.*d.txt
│   │
│   └───glassdoor
│       │   Amazon.glassdoor.csv
│       │   Macys.glassdoor.csv
│       │   Microsoft.glassdoor.csv
│       │   Nordstrom.glassdoor.csv
│       │   collections.glassdoor.csv (renamed the sample csv for unified interface)

```

If you would like to run the classification experiments, please also take a look at the README files in ./OG.MG.RAND_classification and ./SG_classification.
