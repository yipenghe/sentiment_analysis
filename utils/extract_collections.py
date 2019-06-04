from utils.dataLoaders import load_glassdoor

X, y = load_glassdoor("collections", "train", 5, "cons", "sa")
with open("data/fine_tune_docs/pro_from_collection", "w") as f:
   for text in X:
       f.write(text+"\n")
X, y = load_glassdoor("collections", "train", 5, "pros", "sa")
with open("data/fine_tune_docs/con_from_collection", "w") as f:
   for text in X:
       f.write(text+"\n")
X, y = load_glassdoor("collections", "train", 5, "all", "sa")

with open("data/fine_tune_docs/all_from_collection", "w") as f:
   for text in X:
       f.write(text+"\n")
