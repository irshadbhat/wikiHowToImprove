import io
import sys
import random

from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def read_train_test_dev(file_, test_files, dev_files):
    with io.open(file_, encoding='utf-8') as fp:
        train_X = []
        train_y = []
        test_X = []
        test_y = []        
        dev_X = []
        dev_y = []        
        for line in fp:
            title, revision_group, src, tgt = line.strip().split('\t')
            src = src.strip()
            tgt = tgt.strip()
            if len(src.split()) > 150 or len(tgt.split()) > 150:
                continue
            if title in test_files:
                test_X.append(src)
                test_X.append(tgt)
                test_y.append(0)
                test_y.append(1)
            elif title in dev_files:
                dev_X.append(src)
                dev_X.append(tgt)
                dev_y.append(0)
                dev_y.append(1)
            else:
                train_X.append(src)
                train_X.append(tgt)
                train_y.append(0)
                train_y.append(1)
    return train_X, train_y, dev_X, dev_y, test_X, test_y

def train_clf(train_X, train_y, dev_X, dev_y):
    sys.stderr.write('Extracting BOW ...\n')
    sys.stderr.flush()
    count_vect = CountVectorizer(max_features=None, lowercase=False, ngram_range=(1,2), stop_words=None)
    train_X_counts = count_vect.fit_transform(train_X)
    dev_X_counts = count_vect.transform(dev_X)
    print(train_X_counts.shape)
    print(dev_X_counts.shape)
    normalize(train_X_counts, copy=False)
    normalize(dev_X_counts, copy=False)
    sys.stderr.write('BOW feature representation done...\n')
    sys.stderr.write('Training MultinomialNB classifier ...\n')
    sys.stderr.flush()
    clf = MultinomialNB()
    clf.fit(train_X_counts, train_y) 
    sys.stderr.write('Training complete !\n')
    sys.stderr.flush()
    dev_y_ = clf.predict_proba(dev_X_counts)[:, 1]
    good = 0.0
    bad = 0.0
    for i,(s,t) in enumerate(zip(dev_y_[::2], dev_y_[1::2])):
        if s<t:
            good += 1.0
        else:
            bad += 1.0
    print(good/(good+bad))
    sys.stdout.flush()


if __name__ == '__main__':
    dev_files = set(open('dev_files.txt').read().split())
    test_files = set(open('test_files.txt').read().split())
    train_X, train_y, dev_X, dev_y, test_X, test_y = read_train_test_dev(sys.argv[1], test_files, dev_files)
    #train_clf(train_X, train_y, dev_X, dev_y)  # for tuning
    train_clf(train_X, train_y, test_X, test_y)
