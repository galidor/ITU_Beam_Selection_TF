# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:25:40 2020

@author: mehdi
"""
import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import scipy.io as sio
import numpy as np
import tensorflow as tf
import scipy.io as sio
import numpy as np
import scipy.stats
import math
from dataloader import LidarDataset2D

#tf.keras.backend.set_floatx('float64')

MC = 10
NUM_CLIENTS = 2
NUM_EPOCHS = 5
BATCH_SIZE = 32
SHUFFLE_BUFFER = 20
PREFETCH_BUFFER=10
NUM_ROUNDS = 5


lidar_training_path = ["C:/Users/mehdi/Desktop/ITUBeamSelection/ITU-Challenge-ML5G-PHY-master/ITU-Challenge-ML5G-PHY-master/baseline_data/lidar_input/lidar_train.npz",
                       "C:/Users/mehdi/Desktop/ITUBeamSelection/ITU-Challenge-ML5G-PHY-master/ITU-Challenge-ML5G-PHY-master/baseline_data/lidar_input/lidar_validation.npz"]
beam_training_path = ["C:/Users/mehdi/Desktop/ITUBeamSelection/ITU-Challenge-ML5G-PHY-master/ITU-Challenge-ML5G-PHY-master/baseline_data/beam_output/beams_output_train.npz",
                      "C:/Users/mehdi/Desktop/ITUBeamSelection/ITU-Challenge-ML5G-PHY-master/ITU-Challenge-ML5G-PHY-master/baseline_data/beam_output/beams_output_validation.npz"]


lidar_test_path = ["C:/Users/mehdi/Desktop/ITUBeamSelection/ITU-Challenge-ML5G-PHY-master/ITU-Challenge-ML5G-PHY-master/baseline_data/lidar_input/lidar_train.npz",
                       "C:/Users/mehdi/Desktop/ITUBeamSelection/ITU-Challenge-ML5G-PHY-master/ITU-Challenge-ML5G-PHY-master/baseline_data/lidar_input/lidar_validation.npz"]
beam_test_path = ["C:/Users/mehdi/Desktop/ITUBeamSelection/ITU-Challenge-ML5G-PHY-master/ITU-Challenge-ML5G-PHY-master/baseline_data/beam_output/beams_output_train.npz",
                      "C:/Users/mehdi/Desktop/ITUBeamSelection/ITU-Challenge-ML5G-PHY-master/ITU-Challenge-ML5G-PHY-master/baseline_data/beam_output/beams_output_validation.npz"]



def channel(lidar_path, beam_path, num_clients, k):
    training_data = LidarDataset2D(lidar_path, beam_path)
    x = np.expand_dims(training_data.lidar_data, 3)
    print(x.shape)
    xx = x[k*int(x.shape[0]/num_clients):(k+1)*int(x.shape[0]/num_clients),:,:,:]
    print(xx.shape)
    y = training_data.beam_output
    print(y.shape)
    yy = y[k*int(y.shape[0]/num_clients):(k+1)*int(y.shape[0]/num_clients),:]
    print(yy.shape)

    dataset_train = tf.data.Dataset.from_tensor_slices((list(xx.astype(np.float32)),list(yy.astype(np.float32))))

    return dataset_train

def testgen(lidar_path, beam_path):
    test_data = LidarDataset2D(lidar_path, beam_path)
    test_data.lidar_data = np.expand_dims(test_data.lidar_data, 3)
    
    dataset_test = tf.data.Dataset.from_tensor_slices((list(test_data.lidar_data.astype(np.float32)),list(test_data.beam_output.astype(np.float32))))
    return dataset_test

def preprocess(dataset):
  def batch_format_fn(element1,element2):
    return collections.OrderedDict(x=element1, y=element2)

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

temp_dataset = channel(lidar_training_path, beam_training_path,NUM_CLIENTS,0)
preprocessed_example_dataset=preprocess(temp_dataset)
example_element = next(iter((preprocessed_example_dataset)))     



def create_keras_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(20, 200, 1)),
    tf.keras.layers.Conv2D(10, 3, 1, padding='same'),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Conv2D(10, 3, 1, padding='same'),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Conv2D(10, 3, 2, padding='same'),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Conv2D(10, 3, 1, padding='same'),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Conv2D(10, 3, 2, padding='same'),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Conv2D(10, 3, (1, 2), padding='same'),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Softmax()])


def model_fn():
  keras_model = create_keras_model()
  #loss_fn = lambda y_true, y_pred: -tf.reduce_sum(tf.reduce_mean(y_true[y_pred>0] * tf.math.log(y_pred[y_pred>0]), axis=0))
  #top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy', dtype=None)
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(lr=5e-3),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1))


accFL=0  
for MCi in range(MC):

    federated_train_data=[]    
    test_dataset = testgen(lidar_test_path, beam_test_path)  
    
    for i in range(NUM_CLIENTS):
        train_dataset = channel(lidar_training_path, beam_training_path,NUM_CLIENTS,i)
        federated_train_data.append(preprocess(train_dataset))
        
    
    state = iterative_process.initialize()
    state, metrics = iterative_process.next(state, federated_train_data)
    
    
    federated_test_data=[preprocess(test_dataset)]
    evaluation = tff.learning.build_federated_evaluation(model_fn) 
    
    
    for round_num in range(2, NUM_ROUNDS):
      state, metrics = iterative_process.next(state, federated_train_data)
      #test_metrics = evaluation(state.model, federated_test_data)
      print(str(metrics))
    
    #test_metrics = evaluation(state.model, federated_test_data)
    #print(str(test_metrics))

    accFL=accFL+metrics['categorical_accuracy']/MC
    
    print(MCi)
    
    
    
print(accFL)