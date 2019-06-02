from utils.dataLoaders import load_glassdoor

X, y = load_glassdoor("collections", "train", 5, "cons", "sa")
print(len(X))
with open("cons_from_collection", "w") as f:
    for text in X:
        f.write(text+"\n")
