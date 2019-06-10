# Improving Sentiment Classification with Pro/Con Structure of Reviews
This is a course project for CS224U by Di Bai and Yipeng He.

We explored the usage of Pros and Cons structure of reviews, aiming to improve the sentiment classification task. We fine tuned GloVe embedding(https://nlp.stanford.edu/projects/glove/). We release the word embeddings we fine tuned here(https://drive.google.com/drive/u/0/my-drive), please read the readMe.txt inside the folder for usage.
File_structure for data folder (not uploaded because files are large).The dataset we used is also not released. Since we haven't provided the dataset we use, please let us know if you would like to ask for the data to run our experiments.
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
