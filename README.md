# NMF1.0
Use NMF with two constrain to process travel users' comments

Matrix X: tfidf
constrain1: comments' label
constrain2: w2v

1 data preprocessing
2 tfidf and get matrix X
3 use comments' label to build constrain1
4 train word2vec and build constrain2
5 use NMF to classify topics
