# %%
# fetch dataset name list
import tensorflow as tf
import utils.objectives
import utils.metrics
import utils.data_provider
import numpy as np
import os
hhm_path = 'data/hhblits_example/'
pdb_path = 'data/pdb_example/'
hhm_path_files = os.listdir(hhm_path)  
name_list = []
for fi in hhm_path_files: 
    hhm_name = fi.split('.')[0]
    name_list.append(hhm_name)
print(len(name_list))

# # %%
# # one-hot dict for proteins
# import numpy as np
# eyes = np.eye(20)
# protein_dict = {'C':eyes[0], 'D':eyes[1], 'S':eyes[2], 'Q':eyes[3], 'K':eyes[4],
#         'I':eyes[5], 'P':eyes[6], 'T':eyes[7], 'F':eyes[8], 'N':eyes[9],
#         'G':eyes[10], 'H':eyes[11], 'L':eyes[12], 'R':eyes[13], 'W':eyes[14],
#         'A':eyes[15], 'V':eyes[16], 'E':eyes[17], 'Y':eyes[18], 'M':eyes[19]}

# #print(protein_dict)
# # generate sliding window dataset, size = 61 (30+1+30)
# window_length = 61
# shhm_path = 'data/output_example/'
# all_sample_x = []
# all_sample_y = []
# t = int((window_length - 1) / 2)
# with open("data/dataset_alphafold.txt") as file:
#     line = file.readline()
#     while line:
#         label_list = []
#         if(line[0] == '>'):
#             uniprot_id = line[1:].strip()
#             seq = file.readline().strip()
#             label = file.readline().strip()
#             seq_len = len(seq)
#             feature_matrix = np.zeros([seq_len + 2 * t, 52], float) # 20+30+1+1,onehot+spacehhblits+noseq+mask
#             shhm_matrix = np.loadtxt(shhm_path + uniprot_id + ".shhm")
#             # check padding
#             for i in range(feature_matrix.shape[0]):
#                 if(i < t):
#                     feature_matrix[i][-2] = 1 #noseq
#                 elif(i >= seq_len + t):
#                     feature_matrix[i][-2] = 1 #noseq
#                 else:
#                     feature_matrix[i,0:20] = protein_dict[seq[i - t]]
#                     feature_matrix[i,20:50] = shhm_matrix[i - t,:]
#                     feature_matrix[i,-1] = 1 # mask
#         #print(feature_matrix.shape) # seq_len + 2 * t, 52
#         # sliding window
#         top = 0
#         buttom = window_length
#         while(buttom <= feature_matrix.shape[0]):
#             all_sample_x.append(feature_matrix[top:buttom])            
#             top += 1
#             buttom += 1
#         for i in range(seq_len):
#             all_sample_y.append(int(label[i]))
#         #print(np.array(sample_list).shape)
#         #print(np.array(label_list).shape)
#         line = file.readline()        
# all_sample_x = np.array(all_sample_x)
# all_sample_y = np.array(all_sample_y)
# print(all_sample_x.shape)
# print(all_sample_y.shape)
# np.save("data/dataset_5A_simple_average_shhm_slidingwindow_x.npy", all_sample_x)
# np.save("data/dataset_5A_simple_average_shhm_slidingwindow_y.npy", all_sample_y)
# x.shape == (1007491,61,52)

# import numpy as np
# import pandas as pd
# pd.set_option('display.max_columns', None)
# # import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# import os
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.utils import shuffle
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# # %%
# # set input layers
input_feature = tf.keras.layers.Input(shape=[61, 50], name = 'input_feature')
input_mask = tf.keras.layers.Input(shape=[61,], name = 'input_mask')

# # %%
# # 1 test MLP
# # build model
# hidden_1 = tf.keras.layers.Dense(512, activation='relu')(input_feature)
# hidden_2 = tf.keras.layers.Dense(256, activation='relu')(hidden_1)
# drop1 = tf.keras.layers.Dropout(0.3)(hidden_2)
# hidden_3 = tf.keras.layers.BatchNormalization()(drop1)
# hidden_4 = tf.keras.layers.Dense(128, activation='relu')(hidden_3)
# output = tf.keras.layers.Dense(2, activation='softmax', name = 'output_MLP')(hidden_4)
# model_MLP = tf.keras.models.Model(inputs=input_feature, outputs=output)
# model_MLP.summary()
# # # %%
# # # 2 test CNN
# # # build model

hidden_1 = tf.keras.layers.Conv1D(32, 5, kernel_initializer='he_uniform')(input_feature)
hidden_1 = tf.keras.layers.BatchNormalization()(hidden_1)
hidden_1 = tf.keras.layers.Activation('relu')(hidden_1)
hidden_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_1)
hidden_2 = tf.keras.layers.Conv1D(32, 7, kernel_initializer='he_uniform')(hidden_1)
hidden_2 = tf.keras.layers.BatchNormalization()(hidden_2)
hidden_2 = tf.keras.layers.Activation('relu')(hidden_2)
hidden_2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_2)
hidden_3 = tf.keras.layers.Conv1D(32, 7, kernel_initializer='he_uniform')(hidden_2)
hidden_3 = tf.keras.layers.BatchNormalization()(hidden_3)
hidden_3 = tf.keras.layers.Activation('relu')(hidden_3)
hidden_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_3)
hidden_3 = tf.keras.layers.Flatten()(hidden_3)
output = tf.keras.layers.Dense(128, activation='relu')(hidden_3)
output = tf.keras.layers.Dense(2, activation='softmax', name = 'output_CNN')(output)
model_CNN = tf.keras.models.Model(inputs=input_feature, outputs=output)
model_CNN.summary()

# # 3 test RNN
# # build model
# units = 32
# rnn = tf.keras.layers.SimpleRNN(units,return_sequences=True)(input_feature)
# rnn = tf.keras.layers.SimpleRNN(units,return_sequences=True)(rnn)
# rnn = tf.keras.layers.BatchNormalization()(rnn)
# rnn = tf.keras.layers.Flatten()(rnn)
# #print('bet_cnn.get_shape()', rnn.get_shape())
# rnn = tf.keras.layers.Dense(128, activation='relu')(rnn)
# output = tf.keras.layers.Dense(2, activation='softmax', name = 'output_RNN')(rnn)
# model_RNN = tf.keras.models.Model(inputs=input_feature, outputs=output)
# model_RNN.summary()

# # 4 test LSTM
# # build model
# units = 32
# lstm = tf.keras.layers.LSTM(units, return_sequences=True)(input_feature)
# lstm = tf.keras.layers.LSTM(units, return_sequences=True)(lstm)
# lstm = tf.keras.layers.BatchNormalization()(lstm)
# lstm = tf.keras.layers.Flatten()(lstm)
# #print('lstm.get_shape()', lstm.get_shape())
# lstm = tf.keras.layers.Dense(128, activation='relu')(lstm)
# output = tf.keras.layers.Dense(2, activation='softmax', name = 'output_LSTM')(lstm)
# model_LSTM = tf.keras.models.Model(inputs=input_feature, outputs=output)
# model_LSTM.summary()

# # 5 test BiLSTM
# units = 32
# lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(input_feature)
# lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(lstm)
# lstm = tf.keras.layers.BatchNormalization()(lstm)
# lstm = tf.keras.layers.Flatten()(lstm)
# #print('lstm.get_shape()', lstm.get_shape())
# lstm = tf.keras.layers.Dense(128, activation='relu')(lstm)
# output = tf.keras.layers.Dense(2, activation='softmax', name = 'output_BiLSTM')(lstm)
# model_BiLSTM = tf.keras.models.Model(inputs=input_feature, outputs=output)
# model_BiLSTM.summary()
'''
# # 6 test CNN+BiLSTM
units = 700
hidden_1 = tf.keras.layers.Conv1D(32, 5, kernel_initializer='he_uniform')(input_feature)
hidden_1 = tf.keras.layers.BatchNormalization()(hidden_1)
hidden_1 = tf.keras.layers.Activation('relu')(hidden_1)
hidden_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_1)
hidden_2 = tf.keras.layers.Conv1D(32, 7, kernel_initializer='he_uniform')(hidden_1)
hidden_2 = tf.keras.layers.BatchNormalization()(hidden_2)
hidden_2 = tf.keras.layers.Activation('relu')(hidden_2)
hidden_2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_2)
hidden_3 = tf.keras.layers.Conv1D(32, 7, kernel_initializer='he_uniform')(hidden_2)
hidden_3 = tf.keras.layers.BatchNormalization()(hidden_3)
hidden_3 = tf.keras.layers.Activation('relu')(hidden_3)
hidden_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)(hidden_3)
#print('hidden_3.get_shape()', hidden_3.get_shape())
lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(hidden_3)
lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(lstm)
#lstm = tf.keras.layers.BatchNormalization()(lstm)
lstm = tf.keras.layers.Flatten()(lstm)
#print('lstm.get_shape()', lstm.get_shape())
lstm = tf.keras.layers.Dense(128, activation='relu')(lstm)
output = tf.keras.layers.Dense(2, activation='softmax', name = 'output_CNN_BiLSTM')(lstm)
model_CNN_BiLSTM = tf.keras.models.Model(inputs=input_feature, outputs=output)
model_CNN_BiLSTM.summary()
'''
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
# seq_embed_dim = 13
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
# transformer = tf.keras.layers.Dense(128, activation='relu')(transformer)
# output = tf.keras.layers.Dense(2, activation='softmax', name = 'output_Transformer')(transformer)
# model_Transformer = tf.keras.models.Model(inputs=[input_feature,input_mask], outputs=output)
# model_Transformer.summary()



# %%


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
config_vars['batch_size'] = 512
config_vars['learning_rate'] = 7e-4
config_vars['root_directory'] = '/home/panwh/space-hhblits/'
config_vars['learning_rate_decay'] = 0.005

setup_experiment(config_vars, 'test')

x = np.load('/home/panwh/space-hhblits/data/dataset_5A_simple_average_shhm_slidingwindow_x.npy')
y = np.load('/home/panwh/space-hhblits/data/dataset_5A_simple_average_shhm_slidingwindow_y.npy')

seed = np.random.randint(0, 10000)
np.random.seed(seed)
np.random.shuffle(x)
np.random.seed(seed)
np.random.shuffle(y)
# dataxy = np.random.shuffle(np.concatenate([x,np.reshape(y,(len(y),1))],axis=1))
# x = dataxy[0:500000,:,0:20]
# y = dataxy[0:500000]

train_gen = utils.data_provider.DataGenerator(x = x[0:500000,:,0:50],
                                              y = y[0:500000],
                                              batch_size=config_vars['batch_size'],
                                              is_training=True,
                                              shuffle= True)
val_gen = utils.data_provider.DataGenerator(x = x[500000:700000,:,0:50],
                                            y = y[500000:700000],
                                            batch_size=config_vars['batch_size'],
                                            is_training = False,
                                            shuffle = True)

del x
del y
optimizer = tf.keras.optimizers.Adam(lr = config_vars['learning_rate'])
lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: config_vars['learning_rate'] * (config_vars['learning_rate_decay'] ** epoch))
# loss = utils.objectives.weighted_masked_crossentropy(weight=500)
loss = utils.objectives.binary_cross_entropy
metrics = [utils.metrics.sensitivities_metric(func_name = 'sensitivities'),
           utils.metrics.specificities_metric(func_name = 'specificities'),
           utils.metrics.precision_metric(func_name = 'precision'),
           utils.metrics.accuracy_metric(func_name = 'accuracy'),
           utils.metrics.f1_score_metric(func_name = 'f1_score'),
           utils.metrics.mcc_metric(func_name = 'mcc')]
model_CNN.compile(loss=loss, metrics = metrics, optimizer=optimizer)

# model_CNN.compile(loss=loss, optimizer=optimizer)
checkpoint = tf.keras.callbacks.ModelCheckpoint(config_vars["experiment_dir"] + '/weights-{epoch:02d}.h5', monitor='mcc', mode='max', #val_categorical_accuracy val_acc
                                            save_best_only=True, save_weights_only=True, verbose=1) 
# Performance logging
callback_csv = tf.keras.callbacks.CSVLogger(filename=config_vars["csv_log_file"])

callbacks=[checkpoint,callback_csv]
#callbacks=[checkpoint,callback_csv,lr_decay]
model_CNN.fit_generator(
    generator=train_gen,
    # steps_per_epoch=config_vars["steps_per_epoch"],
    epochs=config_vars["epochs"],
    validation_data=val_gen,
    # validation_data = (x[900000:1007941,:,0:50],np.stack([y[900000:1007941],np.logical_not(y[900000:1007941])],axis=1)),
    # validation_steps=int(len(data_partitions["validation"])/config_vars["val_batch_size"]),
    callbacks=callbacks,
    verbose = 1
)
# model_CNN.fit(
#     x = train_set[:,:,0:50],
#     y = train_set[:,:,51:None],
#     batch_size = config_vars['batch_size'],
#     validation_data = (valid_set[:,:,0:50],valid_set[:,:,51:None]),
#     epochs = config_vars['epochs'],
#     callbacks=callbacks,
#     verbose = 1
#     )
model_CNN.save_weights(config_vars["model_file"])
pred = model_CNN.predict(x[900000:1007941,:,0:50])
print(pred)
