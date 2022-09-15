# %%
# fetch dataset name list
from statistics import mode
from utils.Transformer import MultiHeadSelfAttention
from utils.Transformer import TransformerBlock
from utils.Transformer import TokenAndPositionEmbedding
import tensorflow as tf
import utils.objectives
import utils.metrics
import utils.data_provider
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 

class ModelMLPMultiInput():
    def __init__(self) -> None:
        self.model = self._create_model()
    
    @staticmethod
    def model(self):
        return self.model

    def _create_model(self):
        """ create a MLP model with multiple inputs, 
            including features:
                onthot
                hhblits
                svd
                rasa
        Returns:
            model
        """
        # 20+30+20+1
        # 
        input_feature = tf.keras.layers.Input(shape=[31,71],name = 'onehot')
        input_feature_onehot = input_feature[:,:,0:20]
        input_feature_hhblits = input_feature[:,:,20:50]
        # input_feature_svd = input_feature[:,:,50:66]
        # input_feature_rasa = input_feature[:,:,50:51]
        # input_feature_ss = input_feature[:,:,50:53]        


        hidden_input_feature_onehot = tf.keras.layers.Dense(512, activation='relu')(input_feature_onehot)
        hidden_input_feature_hhblits = tf.keras.layers.Dense(512, activation = 'relu')(input_feature_hhblits)
        # hidden_input_feature_svd = tf.keras.layers.Dense(512, activation = 'relu')(input_feature_svd)
        # hidden_input_feature_rasa = tf.keras.layers.Dense(512, activation = 'relu')(input_feature_rasa)
        # hidden_input_feature_ss = tf.keras.layers.Dense(512, activation = 'relu')(input_feature_ss)


        hidden_input_feature_onehot1 = tf.keras.layers.Dense(256, activation='relu')(hidden_input_feature_onehot)
        hidden_input_feature_hhblits1 = tf.keras.layers.Dense(256, activation = 'relu')(hidden_input_feature_hhblits)
        # hidden_input_feature_svd1 = tf.keras.layers.Dense(256, activation = 'relu')(hidden_input_feature_svd)
        # hidden_input_feature_rasa1 = tf.keras.layers.Dense(256, activation = 'relu')(hidden_input_feature_rasa)
        # hidden_input_feature_ss1 = tf.keras.layers.Dense(256, activation = 'relu')(hidden_input_feature_ss)        

        drop_onehot_onehot = tf.keras.layers.Dropout(0.3)(hidden_input_feature_onehot1)
        drop_hhblits = tf.keras.layers.Dropout(0.3)(hidden_input_feature_hhblits1)
        # drop_svd = tf.keras.layers.Dropout(0.3)(hidden_input_feature_svd1)
        # drop_rasa = tf.keras.layers.Dropout(0.3)(hidden_input_feature_rasa1)
        # drop_ss = tf.keras.layers.Dropout(0.3)(hidden_input_feature_ss1)        

        bn_onehot = tf.keras.layers.BatchNormalization()(drop_onehot_onehot)
        bn_hhblits = tf.keras.layers.BatchNormalization()(drop_hhblits)
        # bn_svd = tf.keras.layers.BatchNormalization()(drop_svd)
        # bn_rasa = tf.keras.layers.BatchNormalization()(drop_rasa)
        # bn_ss = tf.keras.layers.BatchNormalization()(drop_ss)        

        hidden_input_feature_onehot2 = tf.keras.layers.Dense(128, activation='relu')(bn_onehot)
        hidden_input_feature_hhblits2 = tf.keras.layers.Dense(128, activation = 'relu')(bn_hhblits)
        # hidden_input_feature_svd2 = tf.keras.layers.Dense(128, activation = 'relu')(bn_svd)
        # hidden_input_feature_rasa2 = tf.keras.layers.Dense(128, activation = 'relu')(bn_rasa)
        # hidden_input_feature_ss2 = tf.keras.layers.Dense(128, activation = 'relu')(bn_ss)        
        
        hidden_4 = tf.keras.layers.Concatenate(axis=-1)([hidden_input_feature_onehot2,
                                                         hidden_input_feature_hhblits2
                                                        #  hidden_input_feature_svd2
                                                        #  hidden_input_feature_rasa
                                                        # hidden_input_feature_ss2
                                                         ])

        hidden_4 = tf.keras.layers.Flatten()(hidden_4)
        output = tf.keras.layers.Dense(2, activation='softmax', name = 'output_MLP')(hidden_4)
        
        model_MLP = tf.keras.models.Model(inputs=input_feature, outputs=output)
        model_MLP.summary()
        return model_MLP

class ModelFactory():

    def __init__(self, model_name) -> None:
        if model_name == 'ModelMLPMultiInput':
            self.model = ModelMLPMultiInput().model
        else:
            raise NotImplementedError


def setup_experiment(experiment_name):
    config_vars = {}
    # training hyperparameters settings
    config_vars['epochs'] = 30
    config_vars['batch_size'] = 1000
    # output files path
    config_vars['learning_rate'] = 1e-2
    config_vars['learning_rate_decay'] = 0.99

    # output settings
    config_vars['root_directory'] = '/home/panwh/space-hhblits/'

    # output dirs
    config_vars["experiment_dir"] = os.path.join(config_vars["root_directory"], "experiments/" + experiment_name + "/")
    os.makedirs(config_vars["experiment_dir"], exist_ok=True)
    # output files path
    config_vars["model_file"] = config_vars["root_directory"] + "experiments/" + experiment_name + "/model.hdf5"
    config_vars["csv_log_file"] = config_vars["root_directory"] + "experiments/" + experiment_name + "/log.csv"

    return config_vars


if __name__ == '__main__':

    # make output dir and set output path
    config_vars = setup_experiment('ModelMLPMultiInput_ligand_onehot_0A_hhblits_4')

    # set model from model factory
    model = ModelFactory('ModelMLPMultiInput').model

    # set training and validation data
    training_x = np.load('/data/pwh/dataset_ligand_0A_simple_average_slidingwindow_training_onehot_hhblits_x.npy')
    training_y = np.load('/data/pwh/dataset_ligand_0A_simple_average_slidingwindow_training_onehot_hhblits_y.npy')
    val_x = np.load('/data/pwh/dataset_ligand_0A_simple_average_slidingwindow_test_onehot_hhblits_x.npy')
    val_y = np.load('/data/pwh/dataset_ligand_0A_simple_average_slidingwindow_test_onehot_hhblits_y.npy')

    train_gen = utils.data_provider.DataGenerator(x = training_x,
                                        y = training_y,
                                        batch_size=config_vars['batch_size'],
                                        shuffle= True)
    val_gen = utils.data_provider.DataGenerator(x = val_x,
                                        y = val_y,
                                        batch_size=config_vars['batch_size'],
                                        shuffle = True)
    optimizer = tf.keras.optimizers.Adam(lr = config_vars['learning_rate'])
    lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: config_vars['learning_rate'] * (config_vars['learning_rate_decay'] ** epoch))
    loss = utils.objectives.binary_cross_entropy
    metrics = [utils.metrics.sensitivities_metric(func_name = 'sensitivities'),
            utils.metrics.specificities_metric(func_name = 'specificities'),
            utils.metrics.precision_metric(func_name = 'precision'),
            utils.metrics.accuracy_metric(func_name = 'accuracy'),
            utils.metrics.f1_score_metric(func_name = 'f1_score'),
            utils.metrics.mcc_metric(func_name = 'mcc')]
    
    model.compile(loss=loss, metrics = metrics, optimizer=optimizer)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(config_vars["experiment_dir"] + '/weights-{epoch:02d}.h5', monitor='val_precision', mode='max', #val_categorical_accuracy val_acc
                                        save_best_only=True, save_weights_only=True, verbose=1) 

    callback_csv = tf.keras.callbacks.CSVLogger(filename=config_vars["csv_log_file"])

    callbacks=[checkpoint, callback_csv, lr_decay]

    class_weight = { 0 : 1.0 , 1 : 3.0 }

    model.fit_generator(
        generator=train_gen,
        epochs=config_vars["epochs"],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose = 1,
        class_weight = class_weight
    )
    model.save_weights(config_vars["model_file"])