#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import options
from fairseq.vectordict import vector_dict

from distributed_train import main as distributed_main
from multiprocessing_train import main as multiprocessing_main
from singleprocess_train import main as singleprocess_main
import numpy as np

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

def main(args):
    if args.distributed_port > 0 \
            or args.distributed_init_method is not None:
        distributed_main(args)
    elif args.distributed_world_size > 1:
        multiprocessing_main(args)
    else:
        singleprocess_main(args)

def load_glove():
    glove_dict = {}
    with open("glove.6B.300d.txt", 'r',encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_dict[word] = vector
    return glove_dict

def load_word2vec():
    word2vec_dict = {}
    wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)
    

if __name__ == '__main__':
    
    # glove
    embeddings = np.load('./glove.npy')
    #embedding_mean = np.mean(embeddings,axis=0)
    #embedding_std = np.std(embeddings,axis=0)
    #vector_dict.embedding = np.nan_to_num((embeddings-embedding_mean)/(embedding_std*3.3))
    vector_dict.embedding = np.nan_to_num(embeddings)
    vector_dict.embedding_dim = 300
    
    # word2vec
    #embeddings = np.load('./word2vec.npy') # load pre_processed word2vec matrix
    #embedding_mean = np.mean(embeddings,axis=0)
    #embedding_std = np.std(embeddings,axis=0)
    #vector_dict.embedding = np.nan_to_num((embeddings-embedding_mean)/(embedding_std*3.3)) # normalization
    #vector_dict.embedding = np.nan_to_num(embeddings)
    #vector_dict.embedding_dim = 500 # setting the embedding dim

    # training 
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
