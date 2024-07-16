# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np

cbow_model = KeyedVectors.load('model/word2vec_CBOW.model')
sg_model = KeyedVectors.load('model/word2vec_skipgram.model')

print('CBOW')
for word in cbow_model.most_similar(u"thủ_tướng"):
    print(word[0])

print('Skip-gram')
for word in sg_model.most_similar(u"thủ_tướng"):
    print(word[0])