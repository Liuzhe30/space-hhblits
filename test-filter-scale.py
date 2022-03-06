# %%
# fetch dataset name list
import os
hhm_path = 'data/hhblits_example/'
pdb_path = 'data/pdb_example/'
hhm_path_files = os.listdir(hhm_path)  
name_list = []
for fi in hhm_path_files: 
    hhm_name = fi.split('.')[0]
    name_list.append(hhm_name)
print(len(name_list))

# %%
# one-hot dict for proteins
import numpy as np
protein_dict = {'C':np.eye(20)[0], 'D':np.eye(20)[1], 'S':np.eye(20)[1], 'Q':np.eye(20)[3], 'K':np.eye(20)[4],
        'I':np.eye(20)[5], 'P':np.eye(20)[6], 'T':np.eye(20)[7], 'F':np.eye(20)[8], 'N':np.eye(20)[9],
        'G':np.eye(20)[10], 'H':np.eye(20)[11], 'L':np.eye(20)[12], 'R':np.eye(20)[13], 'W':np.eye(20)[14],
        'A':np.eye(20)[15], 'V':np.eye(20)[16], 'E':np.eye(20)[17], 'Y':np.eye(20)[18], 'M':np.eye(20)[19]}
#print(protein_dict)

# %%
# generate 1024 cut-off dataset
np.set_printoptions(threshold=np.inf)
# shhm_5A_mae_path = 'data/5A_mae_shhm/'
shhm_5A_mae_path = 'data/output_example/'
all_sample_list = []
with open("data/dataset_alphafold.txt") as file:
    line = file.readline()
    while line:
        if(line[0] == '>'):
            uniprot_id = line[1:].strip()
            seq = file.readline().strip()
            label = file.readline().strip()
            feature_matrix = np.zeros([1024, 52], float) # 20+30+1,onehot+spacehhblits+mask+label
            shhm_matrix = np.loadtxt(shhm_5A_mae_path + uniprot_id + ".shhm")
            # check seq length
            if(len(seq) <= 1024):
                for i in range(len(seq)):
                    feature_matrix[i,0:20] = protein_dict[seq[i]]
                    feature_matrix[i,20:50] = shhm_matrix[i,:]
                    feature_matrix[i,-2] = 1
                    feature_matrix[i,-1] = label[i]
                for i in range(len(seq),1024):
                    feature_matrix[i,-1] = 2 # padding for loss function
            else: # cut off
                for i in range(1024):
                    feature_matrix[i,0:20] = protein_dict[seq[i]]
                    feature_matrix[i,20:50] = shhm_matrix[i,:]
                    feature_matrix[i,-2] = 1
                    feature_matrix[i,-1] = label[i]
        #print(feature_matrix.shape) #1024*52
        all_sample_list.append(feature_matrix)
        line = file.readline()
all_sample = np.array(all_sample_list)
print(all_sample.shape) # (1742, 1024, 52)
np.save("data/dataset_5A_mae_shhm_1024.npy", all_sample)

# %%
# load 1024 cut-off dataset
dataset = np.load("data/dataset_5A_mae_shhm_1024.npy")
print(dataset.shape) # (1742, 1024, 52)
# split dataset
train_set = dataset[0:1642]
valid_set = dataset[1642:1692]
test_set = dataset[1692:1742]

# %%
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# %%
# set input layers
input_feature = tf.keras.layers.Input(shape=[1024, 50], name = 'input_feature')
input_mask = tf.keras.layers.Input(shape=[1024,], name = 'input_mask')

# %%
# 1 test MLP
# build model
# hidden_1 = tf.keras.layers.Dense(512, activation='relu')(input_feature)
# hidden_2 = tf.keras.layers.Dense(256, activation='relu')(hidden_1)
# drop1 = tf.keras.layers.Dropout(0.3)(hidden_2)
# hidden_3 = tf.keras.layers.BatchNormalization()(drop1)
# hidden_4 = tf.keras.layers.Dense(512, activation='relu')(hidden_3)
# output = tf.keras.layers.Dense(1024, activation='relu', name = 'output_MLP')(hidden_4)
# model_MLP = tf.keras.models.Model(inputs=input_feature, outputs=output)
# model_MLP.summary()

# # %%
# # 2 test CNN
# # build model
hidden_1 = tf.keras.layers.Conv1D(32, 5, kernel_initializer='he_uniform')(input_feature)
hidden_1 = tf.keras.layers.BatchNormalization()(hidden_1)
hidden_1 = tf.keras.layers.Activation('relu')(hidden_1)
hidden_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_1)
hidden_2 = tf.keras.layers.Conv1D(32, 7, kernel_initializer='he_uniform')(hidden_1)
hidden_2 = tf.keras.layers.BatchNormalization()(hidden_2)
hidden_2 = tf.keras.layers.Activation('relu')(hidden_2)
hidden_2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_2)
hidden_3 = tf.keras.layers.Conv1D(32, 7, kernel_initializer='he_uniform')(input_feature)
hidden_3 = tf.keras.layers.BatchNormalization()(hidden_3)
hidden_3 = tf.keras.layers.Activation('relu')(hidden_3)
hidden_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_3)
hidden_3 = tf.keras.layers.Flatten()(hidden_3)
output = tf.keras.layers.Dense(2048, activation='relu')(hidden_3)
output = tf.keras.layers.Dense(1024, activation='relu', name = 'output_CNN')(output)
model_CNN = tf.keras.models.Model(inputs=input_feature, outputs=output)
model_CNN.summary()

# # %%
# # 3 test RNN
# # build model
# units = 32
# rnn = tf.keras.layers.SimpleRNN(units,return_sequences=True)(input_feature)
# rnn = tf.keras.layers.SimpleRNN(units,return_sequences=True)(rnn)
# rnn = tf.keras.layers.BatchNormalization()(rnn)
# rnn = tf.keras.layers.Flatten()(rnn)
# #print('bet_cnn.get_shape()', rnn.get_shape())
# rnn = tf.keras.layers.Dense(512, activation='relu')(rnn)
# output = tf.keras.layers.Dense(1024, activation='relu', name = 'output_RNN')(rnn)
# model_RNN = tf.keras.models.Model(inputs=input_feature, outputs=output)
# model_RNN.summary()

# # %%
# # 4 test LSTM
# # build model
# units = 32
# lstm = tf.keras.layers.LSTM(units, return_sequences=True)(input_feature)
# lstm = tf.keras.layers.LSTM(units, return_sequences=True)(lstm)
# lstm = tf.keras.layers.BatchNormalization()(lstm)
# lstm = tf.keras.layers.Flatten()(lstm)
# #print('lstm.get_shape()', lstm.get_shape())
# lstm = tf.keras.layers.Dense(512, activation='relu')(lstm)
# output = tf.keras.layers.Dense(1024, activation='relu', name = 'output_LSTM')(lstm)
# model_LSTM = tf.keras.models.Model(inputs=input_feature, outputs=output)
# model_LSTM.summary()

# # %%
# # 5 test BiLSTM
# units = 32
# lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(input_feature)
# lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(lstm)
# lstm = tf.keras.layers.BatchNormalization()(lstm)
# lstm = tf.keras.layers.Flatten()(lstm)
# #print('lstm.get_shape()', lstm.get_shape())
# lstm = tf.keras.layers.Dense(512, activation='relu')(lstm)
# output = tf.keras.layers.Dense(1024, activation='relu', name = 'output_BiLSTM')(lstm)
# model_BiLSTM = tf.keras.models.Model(inputs=input_feature, outputs=output)
# model_BiLSTM.summary()

# # %%
# # 6 test CNN+BiLSTM
# units = 32
# hidden_1 = tf.keras.layers.Conv1D(32, 5, kernel_initializer='he_uniform')(input_feature)
# hidden_1 = tf.keras.layers.BatchNormalization()(hidden_1)
# hidden_1 = tf.keras.layers.Activation('relu')(hidden_1)
# hidden_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_1)
# hidden_2 = tf.keras.layers.Conv1D(32, 7, kernel_initializer='he_uniform')(hidden_1)
# hidden_2 = tf.keras.layers.BatchNormalization()(hidden_2)
# hidden_2 = tf.keras.layers.Activation('relu')(hidden_2)
# hidden_2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_2)
# hidden_3 = tf.keras.layers.Conv1D(32, 7, kernel_initializer='he_uniform')(input_feature)
# hidden_3 = tf.keras.layers.BatchNormalization()(hidden_3)
# hidden_3 = tf.keras.layers.Activation('relu')(hidden_3)
# hidden_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_3)
# #print('hidden_3.get_shape()', hidden_3.get_shape())
# lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(hidden_3)
# lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(lstm)
# lstm = tf.keras.layers.BatchNormalization()(lstm)
# lstm = tf.keras.layers.Flatten()(lstm)
# #print('lstm.get_shape()', lstm.get_shape())
# lstm = tf.keras.layers.Dense(512, activation='relu')(lstm)
# output = tf.keras.layers.Dense(1024, activation='relu', name = 'output_CNN_BiLSTM')(lstm)
# model_CNN_BiLSTM = tf.keras.models.Model(inputs=input_feature, outputs=output)
# model_CNN_BiLSTM.summary()

# # %%
# # 7 test Transformer
# from utils.Transformer import MultiHeadSelfAttention
# from utils.Transformer import TransformerBlock
# from utils.Transformer import TokenAndPositionEmbedding

# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#     # add extra dimensions to add the padding
#     # to the attention logits.
#     return  seq[:, tf.newaxis, tf.newaxis, :]# (batch_size, 1, 1, seq_len)

# maxlen = 1024
# vocab_size = 5
# embed_dim = 64
# num_heads = 4
# ff_dim = 64
# pos_embed_dim = 64
# seq_embed_dim = 14
# num_heads = 4

# embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim)
# trans_block_1 = TransformerBlock(embed_dim, num_heads, ff_dim)
# trans_block_2 = TransformerBlock(embed_dim, num_heads, ff_dim)

# mask = create_padding_mask(input_mask)
# embedding = embedding_layer([input_mask, input_feature])
# embedding = trans_block_1(embedding, mask)
# embedding = trans_block_2(embedding, mask)
# #print('embedding.get_shape()', embedding.get_shape())

# transformer = tf.keras.layers.Flatten()(embedding)
# transformer = tf.keras.layers.Dense(2048, activation='relu')(transformer)
# output = tf.keras.layers.Dense(1024, activation='relu', name = 'output_Transformer')(transformer)
# model_Transformer = tf.keras.models.Model(inputs=[input_feature,input_mask], outputs=output)
# model_Transformer.summary()

# %%
import utils.objectives
import utils.metrics

def setup_experiment(config_vars, experiment_name):
    # Output dirs
    config_vars["experiment_dir"] = os.path.join(config_vars["root_directory"], "experiments/" + experiment_name + "/")
    # Files
    config_vars["model_file"] = config_vars["root_directory"] + "experiments/" + experiment_name + "/model.hdf5"
    config_vars["csv_log_file"] = config_vars["root_directory"] + "experiments/" + experiment_name + "/log.csv"
    # Make output directories
    os.makedirs(config_vars["experiment_dir"], exist_ok=True)
    return config_vars

config_vars = {}
config_vars['epochs'] = 50
config_vars['batch_size'] = 4
config_vars['learning_rate'] = 1e-3
config_vars['root_directory'] = '/home/panwh/space-hhblits/'


setup_experiment(config_vars, 'test')


optimizer = tf.keras.optimizers.RMSprop(lr = config_vars['learning_rate'])
loss = utils.objectives.weighted_masked_crossentropy(weight=500)
metrics = [utils.metrics.sensitivities_metric(func_name = 'sensitivities'),
           utils.metrics.specificities_metric(func_name = 'specificities'),
           utils.metrics.precision_metric(func_name = 'precision'),
           utils.metrics.accuracy_metric(func_name = 'accuracy'),
           utils.metrics.f1_score_metric(func_name = 'f1_score'),
           utils.metrics.mcc_metric(func_name = 'mcc')]
model_CNN.compile(loss=loss, metrics = metrics, optimizer=optimizer)

# model_CNN.compile(loss=loss, optimizer=optimizer)
checkpoint = keras.callbacks.ModelCheckpoint(config_vars["experiment_dir"] + '/weights-{epoch:02d}.h5', monitor='mcc', mode='max', #val_categorical_accuracy val_acc
                                            save_best_only=True, save_weights_only=True, verbose=1) 
# Performance logging
callback_csv = keras.callbacks.CSVLogger(filename=config_vars["csv_log_file"])

callbacks=[checkpoint,callback_csv]
model_CNN.fit(
    x = train_set[:,:,0:50],
    y = train_set[:,:,51:None],
    batch_size = config_vars['batch_size'],
    validation_data = (valid_set[:,:,0:50],valid_set[:,:,51:None]),
    epochs = config_vars['epochs'],
    callbacks=callbacks,
    verbose = 1
    )
model_CNN.save_weights(config_vars["model_file"])

