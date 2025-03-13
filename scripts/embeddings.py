# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:15:18 2025

@author: borib
"""

#import torch
import math
import os
import numpy as np
from operator import itemgetter
import pandas as pd
import string
import gzip

import numpy as np
#import matplotlib.pyplot as plt

#from gensim.test.utils import common_texts
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api


## Establish the folders
data_fold = '../models/'
results_fold = '../results'
figures_fold = '../figures'

# %%


## Load the hunembed word2vec from file (not on github)
# note: .. means up one folder
file = os.path.join(data_fold, 'word2vec-mnsz2-webcorp_600_w10_n5_i1_m10.w2v')

w2v_model = KeyedVectors.load_word2vec_format(file, binary=False)

# %%



file = os.path.join(data_fold, 'numberbatch-19.08.txt.gz')

conceptnet = api.load("conceptnet-numberbatch-17-06-300")

file_path = os.path.join(data_fold, 'numberbatch-19.08.txt.gz')

with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    conceptnet = KeyedVectors.load_word2vec_format(f, binary=False)
