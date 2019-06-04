from collections import Counter
import stanfordnlp
nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma')

def read_doc(path, term_freq=10):
    all_docs = []
    high_freq_words = []
    low_freq_words = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.strip().split(" ")
            all_docs.extend(splitted)
    word_counts = Counter(all_docs)
    for word in word_counts:
        if word_counts[word] < term_freq:
            low_freq_words.append(word)
        else:
            high_freq_words.append(word)
    print(len(high_freq_words))
    print(len(low_freq_words))
    lemma_freq = []
    lemma_dict = {}
    true_low_word = []
    stop_word = []
    low_freq_words = "\n\n".join(low_freq_words)
    doc = nlp(low_freq_words)
    for index,sent in enumerate(doc.sentences):
        if(index%1000 == 0):
            print("completed", index)
        word = sent.words[0]
        lemma = word.lemma
        if (word_counts[lemma] >= 10):
            lemma_dict[word.text] = lemma
        else:
            stop_word.append(word.text)
    high_freq_set = set(high_freq_words)
    stop_word_set = set(stop_word)
    print(len(high_freq_set))
    print(len(stop_word_set))
    print(len(lemma_dict))
    print("=====================")
    return high_freq_set, lemma_dict, stop_word_set

def replace_with_lemma(path, file_type, high_freq_set, lemma_dict, stop_word_set):
    with open(path) as f:
        with open("data/fine_tune_docs/"+file_type + "_lemma_processed_doc", "w") as lemma_file:
            lines = f.readlines()
            for line in lines:
                splitted = line.strip().strip("\n").split()
                for index, word in enumerate(splitted):
                    if word in high_freq_set or word in stop_word_set:
                        #remain the same
                        continue
                    elif word in lemma_dict:
                        splitted[index] = lemma_dict[word]
                new_line = " ".join(splitted)
                lemma_file.write(new_line+'\n')
    with open("../data/fine_tune_docs/"+file_type+"_stop_words", "w") as f:
        for stop_word in stop_word_set:
            f.write(stop_word+"\n")
        #print(doc.sentences[0].words[0].lemma)
freq, lemma, stop = read_doc("../data/fine_tune_docs/pros_from_collection")
replace_with_lemma("../data/fine_tune_docs/pros_from_collection", "pro", freq, lemma, stop)
freq, lemma, stop = read_doc("../data/fine_tune_docs/data/fine_tune_docs/cons_from_collection")
replace_with_lemma("../data/fine_tune_docs/cons_from_collection", "con", freq, lemma, stop)
freq, lemma, stop = read_doc("../data/fine_tune_docs/all_from_collection")
replace_with_lemma("../data/fine_tune_docs/all_from_collection", "all", freq, lemma, stop)