wikiHowToImprove
================

wikiHowToImprove: A Resource and Analyses on Edits in Instructional Texts

Dependencies
------------

  - pip install -r requirements.txt
  - install dynet with GPU support: 

    - BACKEND=cuda pip install git+https://github.com/clab/dynet#egg=dynet
  - get wikiHowToImprove corpus:

    - wget https://bitbucket.org/irshadbhat/wikihowtoimprove-corpus/raw/e76ebb974beb5ec859ebb9f5c78037b80c45e42c/wikiHow_revisions_corpus.txt.bz2
    - bunzip2 wikiHow_revisions_corpus.txt.bz2

  - get fastText embeddings:

    - wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
    - gunzip cc.en.300.vec.gz

Train models from scratch
-------------------------

  - python baseline_bow_clf.py wikiHow_revisions_corpus.txt
  - python lstm_binary_clf.py --data_file wikiHow_revisions_corpus.txt --test_files data/test_files.txt --dev_files data/dev_files.txt --pre_word_vec cc.en.300.vec  --bin_vec 0 --save models/lstm_clf_model --batch_size 256 --dynet-devices CPU,GPU:0 --iter 25
  - python lstm_pairwise_ranking.py --data_file wikiHow_revisions_corpus.txt --test_files data/test_files.txt --dev_files data/dev_files.txt --pre_word_vec cc.en.300.vec  --bin_vec 0 --save models/lstm_ranking_model --batch_size 256 --dynet-devices CPU,GPU:0 --iter 25

Reproduce results with pre-trained models
-----------------------------------------

  - python baseline_bow_clf.py wikiHow_revisions_corpus.txt
  - python lstm_binary_clf.py --data_file wikiHow_revisions_corpus.txt --test_files data/test_files.txt --dev_files data/dev_files.txt --pre_word_vec cc.en.300.vec  --bin_vec 0 --load models/lstm_clf_model --batch_size 256 --dynet-devices CPU,GPU:0  
  - python lstm_pairwise_ranking.py --data_file wikiHow_revisions_corpus.txt --test_files data/test_files.txt --dev_files data/dev_files.txt --pre_word_vec cc.en.300.vec  --bin_vec 0 --load models/lstm_ranking_model --batch_size 256 --dynet-devices CPU,GPU:0
