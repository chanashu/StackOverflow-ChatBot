starspace train -trainFile models/prepared_corpus.tsv -model models/word_embeddings -trainMode 3  -adagrad true -ngrams 1 -similarity cosine  -epoch 5 -dim 100 -minCount 2 -verbose true -fileFormat labelDoc -negSearchLimit 10 -lr 0.05
