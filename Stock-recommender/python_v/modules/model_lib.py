import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
from sklearn.model_selection import train_test_split
import time
import os

__all__=['train','save_loss_map','load_from_h5']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

def get_inputs():
    u_id = tf.keras.layers.Input(shape=(1,), dtype='int32', name='u_id')  
    u_attentions = tf.keras.layers.Input(shape=(1,), dtype='int32', name='u_attentions')
    u_total_clicks = tf.keras.layers.Input(shape=(1,), dtype='int32', name='u_total_clicks')
    s_id = tf.keras.layers.Input(shape=(1,), dtype='int32', name='s_id') 
    m_id = tf.keras.layers.Input(shape=(1,), dtype='int32', name='m_id') 
    s_strategyList = tf.keras.layers.Input(shape=(40,), dtype='int32', name='s_strategyList') 
    s_attentions = tf.keras.layers.Input(shape=(1,), dtype='int32', name='s_attentions') 
    s_total_clicks = tf.keras.layers.Input(shape=(1,), dtype='int32', name='s_total_clicks') 
    
    return u_id,u_attentions,u_total_clicks,s_id, m_id, s_strategyList,s_attentions,s_total_clicks

def get_user_embedding(u_id):
    u_id_embed_layer = tf.keras.layers.Embedding(uid_max, embed_dim, input_length=1, name='u_id_embed_layer')(u_id)
    
    return u_id_embed_layer

def get_user_feature_layer(u_id_embed_layer,u_attentions,u_total_clicks):
    #
    u_id_fc_layer = tf.keras.layers.Dense(embed_dim, name="u_id_fc_layer", activation='relu')(u_id_embed_layer)
    u_attentions_fc_layer = tf.keras.layers.Dense(embed_dim, name="u_attentions_fc_layer", activation='relu')(u_attentions)
    u_total_clicks_fc_layer = tf.keras.layers.Dense(embed_dim, name="u_total_clicks_fc_layer", activation='relu')(u_total_clicks)
    #
    u_id_fc_layer = tf.keras.layers.Reshape([1 * 32])(u_id_fc_layer)
    user_combine_layer = tf.keras.layers.concatenate([u_id_fc_layer,u_attentions_fc_layer,u_total_clicks_fc_layer])
    user_combine_layer = tf.keras.layers.Dense(400, activation='relu')(user_combine_layer)

    user_combine_layer_flat = tf.keras.layers.Reshape([400], name="user_combine_layer_flat")(user_combine_layer)
    return user_combine_layer, user_combine_layer_flat

def get_stock_embedding(s_id, m_id):
    s_id_embed_layer = tf.keras.layers.Embedding(sid_max, embed_dim, input_length=1, name='stock_id_embed_layer')(s_id)
    m_id_embed_layer = tf.keras.layers.Embedding(mid_max, embed_dim // 2, input_length=1, name='stock_mkID_embed_layer')(m_id)
    
    return s_id_embed_layer,m_id_embed_layer

def get_stock_strategies_layers(s_strategyList):
    s_strategyList_embed_layer = tf.keras.layers.Embedding(ssid_max, embed_dim*2, input_length=40, name='s_strategyList_embed_layer')(s_strategyList)
    s_strategyList_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(s_strategyList_embed_layer)

    return s_strategyList_embed_layer

def get_stock_feature_layer(s_id_embed_layer,m_id_embed_layer,s_attentions,s_total_clicks,s_strategyList_embed_layer):
    #
    s_id_fc_layer = tf.keras.layers.Dense(embed_dim, name="stock_id_fc_layer", activation='relu')(s_id_embed_layer)
    m_id_fc_layer = tf.keras.layers.Dense(embed_dim, name="stock_mkID_fc_layer", activation='relu')(m_id_embed_layer)
    s_attentions_fc_layer = tf.keras.layers.Dense(embed_dim, name="s_attentions_fc_layer", activation='relu')(s_attentions)
    s_total_clicks_fc_layer = tf.keras.layers.Dense(embed_dim, name="s_total_clicks_fc_layer", activation='relu')(s_total_clicks)
    #
    s_strategyList_fc_layer = tf.keras.layers.Dense(embed_dim*2, name="s_strategyList_fc_layer", activation='relu')(s_strategyList_embed_layer)
    #
    s_id_fc_layer = tf.keras.layers.Reshape([1 * 32])(s_id_fc_layer)
    m_id_fc_layer = tf.keras.layers.Reshape([1 * 32])(m_id_fc_layer)
    s_strategyList_fc_layer = tf.keras.layers.Reshape([1 * 64])(s_strategyList_fc_layer)
    stock_combine_layer = tf.keras.layers.concatenate([s_id_fc_layer,m_id_fc_layer,s_strategyList_fc_layer,s_attentions_fc_layer, s_total_clicks_fc_layer])
    stock_combine_layer = tf.keras.layers.Dense(400, activation='relu')(stock_combine_layer)
    
    stock_combine_layer_flat = tf.keras.layers.Reshape([400], name="stock_combine_layer_flat")(stock_combine_layer)
    return stock_combine_layer, stock_combine_layer_flat

def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]

class mv_network(object):
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.best_loss = 9999
        self.losses = {'train': [], 'test': [],'epo_train':[],'epo_test':[]}
        #輸入
        u_id,u_attentions,u_total_clicks,s_id, m_id, s_strategyList,s_attentions,s_total_clicks\
            = get_inputs()
        # User嵌入向量
        u_id_embed_layer = get_user_embedding(u_id)
        # User特徵
        user_combine_layer,user_combine_layer_flat \
            = get_user_feature_layer(u_id_embed_layer,u_attentions,u_total_clicks)
        #  stock嵌入向量
        s_id_embed_layer,m_id_embed_layer\
            =get_stock_embedding(s_id, m_id)
        # Stock strategies嵌入向量
        s_strategyList_embed_layer \
            = get_stock_strategies_layers(s_strategyList)            
        # Stock特徵
        stock_combine_layer, stock_combine_layer_flat \
            = get_stock_feature_layer(s_id_embed_layer,m_id_embed_layer,s_attentions,s_total_clicks,s_strategyList_embed_layer)
        
        # 評分
        # 相乘(UxD)
        # inference = tf.keras.layers.Lambda(lambda layer: 
        #     tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")((user_combine_layer_flat, stock_combine_layer_flat))
        # inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)
        # 全連接(UfcS)
        inference_layer = tf.keras.layers.concatenate([user_combine_layer_flat, stock_combine_layer_flat],1)
        inference_dense = tf.keras.layers.Dense(64, kernel_regularizer=tf.nn.l2_loss, activation='relu')(inference_layer)
        inference = tf.keras.layers.Dense(1, name="inference")(inference_dense)

        self.model = tf.keras.Model(
            inputs=[u_id,u_attentions,u_total_clicks,s_id, m_id,s_attentions,s_total_clicks,s_strategyList],
            outputs=[inference])
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # MSE回歸評分
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()

    def compute_loss(self, labels, logits):
#         return tf.reduce_mean(tf.keras.losses.mse(labels, logits))
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

    def compute_metrics(self, labels, logits):
        return tf.keras.metrics.mae(labels, logits)  #

    @tf.function
    def train_step(self, x, y):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        #         metrics = 0
        with tf.GradientTape() as tape:
            logits = self.model([x[0],
                                 x[1],
                                 x[2],
                                 x[3],
                                 x[4],
                                 x[5],
                                 x[6],
                                 x[7]],training=True)
            loss = self.ComputeLoss(y, logits)
            # loss = self.compute_loss(labels, logits)
            self.ComputeMetrics(y, logits)
            # metrics = self.compute_metrics(labels, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits

    def training(self, features, targets_values, epochs=5, log_freq=50):
        for epoch_i in range(epochs):
            train_X, test_X, train_y, test_y = train_test_split(features,
                                                                targets_values,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = get_batches(train_X, train_y, self.batch_size)
            batch_num = (len(train_X) // self.batch_size)

            train_start = time.time()
            # with self.train_summary_writer.as_default():
            if True:
                start = time.time()
                # Metrics are stateful. They accumulate values and return a cumulative
                # result when you call .result(). Clear accumulated values with .reset_states()
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
                # avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

                # Datasets can be iterated over like any other Python iterable.
                for batch_i in range(batch_num):
                    x, y = next(train_batches)
                    strategies = np.zeros([self.batch_size, 40])
                    for i in range(self.batch_size):
                        strategies[i] = x.take(7, 1)[i]
                    loss, logits = self.train_step([np.reshape(x.take(0, 1),[self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(2, 1),[self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(3, 1),[self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(1, 1),[self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(4, 1),[self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(5, 1),[self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(6, 1),[self.batch_size, 1]).astype(np.float32),
                                                    strategies.astype(np.float32)],
                                                   np.reshape(y, [self.batch_size, 1]).astype(np.float32))
                    avg_loss(loss)
                    # avg_mae(metrics)
                    self.losses['train'].append(loss)

                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        # summary_ops_v2.scalar('loss', avg_loss.result(), step=self.optimizer.iterations)
                        # summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=self.optimizer.iterations)
                        # summary_ops_v2.scalar('mae', avg_mae.result(), step=self.optimizer.iterations)
                        rate = log_freq / (time.time() - start)
                        print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(),
                            epoch_i,
                            batch_i,
                            batch_num,
                            loss, (self.ComputeMetrics.result()), rate))
                        # print('Step #{}\tLoss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                        # self.optimizer.iterations.numpy(), loss, (avg_mae.result()), rate))
                        avg_loss.reset_states()
                        self.ComputeMetrics.reset_states()
                        # avg_mae.reset_states()
                        start = time.time()

            train_end = time.time()
            self.losses['epo_train'].append(sum(self.losses['train'][-batch_num:])/batch_num)
            print('\nTrain time for epoch #{} ({} total steps): {}'\
                    .format(epoch_i + 1, self.optimizer.iterations.numpy(),train_end - train_start))
            # with self.test_summary_writer.as_default():
            self.testing((test_X, test_y), self.optimizer.iterations)
    
    def save_model(self,VERSION):
        self.model.save('./saves/'+str(VERSION)+'/model/model.h5')

    def load_saves(self,filename):
        self.model = tf.keras.models.load_model(filename)
            
    def testing(self, test_dataset, step_num):
        test_X, test_y = test_dataset
        test_batches = get_batches(test_X, test_y, self.batch_size)

        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        #         avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

        batch_num = (len(test_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)
            strategies = np.zeros([self.batch_size, 40])
            for i in range(self.batch_size):
                strategies[i] = x.take(7, 1)[i]

            logits = self.model([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(5, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(6, 1), [self.batch_size, 1]).astype(np.float32),
                                 strategies.astype(np.float32)],training=False)
            test_loss = self.ComputeLoss(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            avg_loss(test_loss)
            # 保存测试损失
            self.losses['test'].append(test_loss)
            self.ComputeMetrics(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            # avg_loss(self.compute_loss(labels, logits))
            # avg_mae(self.compute_metrics(labels, logits))
        self.losses['epo_test'].append(sum(self.losses['test'][-batch_num:])/batch_num)
        print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), self.ComputeMetrics.result()))
        # print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), avg_mae.result()))
        #         summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
        #         summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=step_num)
        # summary_ops_v2.scalar('mae', avg_mae.result(), step=step_num)

        if avg_loss.result() < self.best_loss:
            self.best_loss = avg_loss.result()
            print("best loss = {}".format(self.best_loss))
    
    def forward(self, xs):
        predictions = self.model(xs)
        # logits = tf.nn.softmax(predictions)

        return predictions

def declaration_variables(features,strategyID_map):
    global uid_max,sid_max,mid_max,ssid_max
    uid_max = max(features.take(0,1)) + 1
    sid_max = max(features.take(1,1))+ 1
    mid_max = max(features.take(4,1))+ 1
    ssid_max = max(strategyID_map.values()) + 1
    global embed_dim,batch_size,learning_rate
    embed_dim = 32
    batch_size = 256
    learning_rate = 0.0001

def train(features, targets_values,strategyID_map,VERSION, epochs=5):
    declaration_variables(features,strategyID_map)
    mv_net=mv_network()
    mv_net.training(features, targets_values,epochs)
    mv_net.save_model(VERSION)
    return mv_net

def load_from_h5(filename,features,strategyID_map):
    declaration_variables(features,strategyID_map)
    mv_net=mv_network()
    mv_net.load_saves(filename) 
    return mv_net

def save_loss_map(mv_net,lable,VERSION):
    plt.plot(mv_net.losses[lable], label=lable+' loss')
    loss=round(float(mv_net.best_loss), 3)
    plt.axhline(y=loss, color="red",label='best_test_loss='+str(loss))    
    plt.legend()
    _ = plt.ylim()
    plt.savefig('./saves/'+str(VERSION)+'/loss_fig/'+lable+' loss.png',dpi=1000)
    # plt.show()
    plt.close()