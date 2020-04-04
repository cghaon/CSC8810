"""

@author: Jun
"""


import tensorflow as tf
import numpy as np
from dataset import dataset
import pickle
import math

inputNum = 4096
hiddenNum = 256
frameNum = 80
batchSize = 32
epochNum = 50
class S2VT:
    def __init__(self, inputNum, hiddenNum, frameNum = 0, max_caption_len = 50, lr = 0.001, sampling = 0.8):
        self.inputNum = inputNum
        self.hiddenNum = hiddenNum
        self.frameNum = frameNum
        self.max_caption_len = max_caption_len
        self.learning_rate = lr
        self.sampling_prob = sampling
        self.saver = None
        self.vocab_num = None
        self.token = None
        
    def load_vocab(self):
        with open('tokenizer.pickle', 'rb') as handle:
            self.token = pickle.load(handle)
        self.vocab_num = len(self.token.word_index)
            
    def build_model(self, feat, cap, cap_len, isTrain=True):
        W_top = tf.Variable(tf.random_uniform([self.inputNum, self.hiddenNum],-0.1,0.1), name='W_top')
        b_top = tf.Variable(tf.zeros([self.hiddenNum]), name='b_top')
        W_btm = tf.Variable(tf.random_uniform([self.hiddenNum,self.vocab_num],-0.1,0.1), name='W_btm')
        b_btm = tf.Variable(tf.zeros([self.vocab_num]),name='b_btm')
        embedding = tf.Variable(tf.random_uniform([self.vocab_num,self.hiddenNum],-0.1,0.1), name='Embedding')
        batchSize = tf.shape(feat)[0]
        
        with tf.variable_scope('LSTMTop'):
            lstm_top = tf.nn.rnn_cell.BasicLSTMCell(self.hiddenNum, forget_bias=1.0, state_is_tuple=True)
            if isTrain:
                lstm_top = tf.contrib.rnn.DropoutWrapper(lstm_top, output_keep_prob=0.5)    
        with tf.variable_scope('LSTMBottom'):
            lstm_btm = tf.nn.rnn_cell.BasicLSTMCell(self.hiddenNum, forget_bias=1.0, state_is_tuple=True)
            if isTrain:
                lstm_btm = tf.contrib.rnn.DropoutWrapper(lstm_btm, output_keep_prob=0.5)
                
        if isTrain:
            cap_mask = tf.sequence_mask(cap_len,self.max_caption_len, dtype=tf.float32)
            feat = tf.nn.dropout(feat,0.5)
        feat = tf.reshape(feat,[-1,self.inputNum])
        img_emb = tf.add(tf.matmul(feat,W_top),b_top)
        img_emb = tf.transpose(tf.reshape(img_emb,[-1, self.frameNum, self.hiddenNum]),perm=[1,0,2])
                
        h_top = lstm_top.zero_state(batchSize, dtype=tf.float32)
        h_btm = lstm_top.zero_state(batchSize, dtype=tf.float32)
        
        pad = tf.ones([batchSize, self.hiddenNum])*self.token.texts_to_sequences(['<PAD>'])[0][0]
        
        for i in range(frameNum):
            with tf.variable_scope('LSTMTop'):
                output_top, h_top = lstm_top(img_emb[i,:,:],h_top)
            with tf.variable_scope('LSTMBottom'):
                output_btm, h_btm = lstm_btm(tf.concat([pad,output_top],axis=1),h_top)
                
        logit = None
        logit_list = []
        cross_entropy_list = []
     
        for i in range(0, self.max_caption_len):
            with tf.variable_scope('LSTMTop'):
                output_top, h_top = lstm_top(pad, h_top)

            if i == 0:
                with tf.variable_scope('LSTMBottom'):
                    bos = tf.ones([batchSize, self.hiddenNum])*self.token.texts_to_sequences(['<BOS>'])[0][0]
                    bos_btm_input = tf.concat([bos, output_top], axis=1)
                    output_btm, h_btm = lstm_btm(bos_btm_input, h_btm)
            else:
                if isTrain:
                    if np.random.uniform(0,1,1) < self.sampling_prob:
                        input_btm = cap[:,i-1]
                    else:
                        input_btm = tf.argmax(logit, 1)
                else:
                    input_btm = tf.argmax(logit, 1)
                btm_emb = tf.nn.embedding_lookup(embedding, input_btm)
                with tf.variable_scope('LSTMBottom'):
                    input_btm_emb = tf.concat([btm_emb, output_top], axis=1)
                    output_btm, h_btm = lstm_btm(input_btm_emb, h_btm)
                    
            logit = tf.add(tf.matmul(output_btm, W_btm), b_btm)
            logit_list.append(logit)
            
            if isTrain:
                labels = cap[:, i]
                one_hot_labels = tf.one_hot(labels, self.vocab_num, on_value = 1, off_value = None, axis = 1) 
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=one_hot_labels)
                cross_entropy = cross_entropy * cap_mask[:, i]
                cross_entropy_list.append(cross_entropy)
        
        if isTrain:
            cross_entropy_list = tf.stack(cross_entropy_list, 1)
            loss = tf.reduce_sum(cross_entropy_list, axis=1)
            loss = tf.divide(loss, tf.cast(cap_len, tf.float32))
            loss = tf.reduce_mean(loss, axis=0)

        logit_list = tf.stack(logit_list, axis = 0)
        logit_list = tf.reshape(logit_list, (self.max_caption_len, batchSize, self.vocab_num))
        logit_list = tf.transpose(logit_list, [1, 0, 2])
        if isTrain:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = optimizer.minimize(loss)
        else:
            train_op = None
            loss = None
        pred_op = tf.argmax(logit_list, axis=2)
        return train_op, loss, pred_op, logit_list

def train():
    data_train = dataset(batchSize,'./MLDS_hw2_1_data/training_data/feat/','./MLDS_hw2_1_data/training_label.json')
    data_train.generate_token()
    data_train.process_data()
    data_train.save_vocab()
    graph_train = tf.Graph()
    with graph_train.as_default():
        model = S2VT(inputNum,hiddenNum,frameNum)
        model.load_vocab()
        feat = tf.placeholder(tf.float32, [None, frameNum, inputNum], name='features')
        cap = tf.placeholder(tf.int32, [None, 50], name='caption')
        cap_len = tf.placeholder(tf.int32, [None], name='captionLength')
        train_op, loss_op, pred_op, logit_list_op = model.build_model(feat, cap, cap_len, True)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=3)
    sess_train = tf.Session(graph=graph_train)
    training_loss = []
    num_step = int(data_train.data_size/batchSize)
    sess_train.run(init)
    for epoch in range(epochNum):
        data_train.shuffle()
        for i in range(num_step):
            id_batch, feat_batch, cap_batch, cap_len_batch,  = data_train.next_batch()
            sess_train.run(train_op,feed_dict={feat:feat_batch,cap:cap_batch,cap_len:cap_len_batch})
        loss = sess_train.run(loss_op,feed_dict={feat:feat_batch,cap:cap_batch,cap_len:cap_len_batch})
        training_loss.append(loss)
        print("Epoch: ",epoch," Loss: ",loss)
    model_path = saver.save(sess_train, './save/model', global_step=epochNum*num_step)
    print("Model saved: ",model_path)
    
    
def test():
    data_test = dataset(batchSize,'./MLDS_hw2_1_data/testing_data/feat/')
    data_test.load_token()
    data_test.process_train_data()

    test_graph = tf.Graph()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with test_graph.as_default():
        model = S2VT(inputNum,hiddenNum,frameNum)
        model.load_vocab()
        feat = tf.placeholder(tf.float32, [None, frameNum, inputNum], name='features')
        cap = tf.placeholder(tf.int32, [None, 50], name='caption')
        cap_len = tf.placeholder(tf.int32, [None], name='captionLength')
        _, _, pred_op, logit_list_op = model.build_model(feat, cap, cap_len, isTrain=False)
        saver = tf.train.Saver(max_to_keep=3)

    sess = tf.Session(graph=test_graph, config=gpu_config)
    latest_checkpoint = tf.train.latest_checkpoint("./save/")    
    saver.restore(sess, latest_checkpoint)
    txt = open('./MLDS_hw2_1_data/hw2Result', 'w')
    num_steps = int(math.ceil(data_test.dataSize/batchSize))
    eos = model.token.texts_to_sequences(['<EOS>'])[0][0]
    eosIndex = model.max_caption_len
    for i in range(num_steps):
        id_batch, feat_batch = data_test.next_train_batch()
        prediction = sess.run(pred_op,feed_dict={feat:feat_batch})
        for j in range(len(feat_batch)):
            for k in range(model.max_caption_len):
                if prediction[j][k]== eos:
                    eosIndex = k
                    break
            cap_output = model.token.sequences_to_texts([prediction[j][0:eosIndex]])[0]
            txt.write(id_batch[j] + "," + str(cap_output) + "\n")
    txt.close()

    
    
if __name__ == '__main__':
    test()
    
    