# -*- coding: utf-8 -*-
"""

@author: Jun Xiang
"""
from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence
import numpy as np
import json
import pickle
import os

class dataset():
    
    def __init__(self,batchSize,featDir,labelDir = None,maxClen=50):
        self.batchSize = batchSize
        self.featDir = featDir
        self.labelDir = labelDir
        self.maxClen = maxClen
        self.batchIndex = 0
        self.vocabSize = 0
        self.dataSize = None
        self.dataIndex = None
        self.idList = None
        self.capList = None
        self.capLengthList = None
    
    def generate_token(self):
        self.tokenizer = Tokenizer(filters='`","?!/.()',split=" ")
        total_list = []
        with open(self.labelDir) as f:
            rawData = json.load(f)
        for vid in rawData:
            for cap in vid['caption']:
                total_list.append(cap)
        self.tokenizer.fit_on_texts(total_list)
        self.vocabSize = len(self.tokenizer.word_index)
        self.tokenizer.fit_on_texts(['<PAD>','<BOS>','<EOS>','<UNK>'])
    
    def load_token(self):
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.vocabSize = len(self.tokenizer.word_index) - 4
        
    def process_data(self):
        
        pad = self.tokenizer.texts_to_sequences(['<PAD>'])[0]
        idList = []
        capList = []
        capLengthList = []
        self.feat_data = {}
        with open(self.labelDir) as f:
            rawData = json.load(f)
        for vid in rawData:
            vid_id = vid['id']
            self.feat_data[vid_id] = np.load(self.featDir + vid_id + '.npy')

            for caption in vid['caption']:
                words = text_to_word_sequence(caption)
                for i in range(len(words)):
                    if words[i] not in self.tokenizer.word_index:
                        words[i] = '<UNK>'
                words.append('<EOS>')
                one_hot = self.tokenizer.texts_to_sequences([words])[0]
                cap_length = len(one_hot)
                one_hot += pad * (self.maxClen - cap_length)
                idList.append(vid_id)
                capList.append(one_hot)
                capLengthList.append(cap_length)
                
        self.idList = np.array(idList)
        self.capList = np.array(capList)
        self.capLengthList = np.array(capLengthList)
        self.dataSize = len(self.capList)
        self.dataIndex = np.arange(self.dataSize,dtype=np.int)
    def process_train_data(self):
        idList = []
        self.feat_data = {}
        for filename in os.listdir(self.featDir):
            if filename.endswith('.npy'):
                vid_id = os.path.splitext(filename)[0]
                self.feat_data[vid_id] = np.load(self.featDir + filename)
                idList.append(vid_id)
            self.idList = np.array(idList)
        self.dataSize = len(self.idList)
        self.dataIndex = np.arange(self.dataSize,dtype=np.int) 
        
    def save_vocab(self):
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def shuffle(self):
        np.random.shuffle(self.dataIndex)
        
    def next_batch(self):
        if self.batchIndex + self.batchSize <= self.dataSize:
            idx = self.dataIndex[self.batchIndex:(self.batchIndex+self.batchSize)]
            self.batchIndex += self.batchSize
        else:
            idx = self.dataIndex[self.batchIndex:]
            self.batchIndex = 0
        id_batch = self.idList[idx]
        cap_batch = self.capList[idx]
        cap_length_batch = self.capLengthList[idx]
        feat_batch = []
        for vid_id in id_batch:
            feat_batch.append(self.feat_data[vid_id])
        feat_batch = np.array(feat_batch)
        return id_batch,feat_batch,cap_batch,cap_length_batch

    def next_train_batch(self):
        if self.batchIndex + self.batchSize <= self.dataSize:
            idx = self.dataIndex[self.batchIndex:(self.batchIndex+self.batchSize)]
            self.batchIndex += self.batchSize
        else:
            idx = self.dataIndex[self.batchIndex:]
            self.batchIndex = 0
        id_batch = self.idList[idx]
        feat_batch = []
        for vid_id in id_batch:
            feat_batch.append(self.feat_data[vid_id])
        feat_batch = np.array(feat_batch)
        return id_batch,feat_batch
