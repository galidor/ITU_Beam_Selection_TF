import sys

import tensorflow as tf
from tensorflow.keras import losses, optimizers
import numpy as np

from dataloader import LidarDataset2D, LidarDataset3D
from models import Lidar2D, LidarMarcus

if __name__ == '__main__':
    lidar_training_path = ["/home/galidor/Documents/ITU_Beam_Selection/data/baseline_data/lidar_input/lidar_train.npz",
                           "/home/galidor/Documents/ITU_Beam_Selection/data/baseline_data/lidar_input/lidar_validation.npz"]
    beam_training_path = ["/home/galidor/Documents/ITU_Beam_Selection/data/baseline_data/beam_output/beams_output_train.npz",
                          "/home/galidor/Documents/ITU_Beam_Selection/data/baseline_data/beam_output/beams_output_validation.npz"]
    training_data = LidarDataset3D(lidar_training_path, beam_training_path)

    lidar_test_path = "/home/galidor/Documents/ITU_Beam_Selection/data/baseline_data/lidar_input/lidar_test.npz"
    beam_test_path = "/home/galidor/Documents/ITU_Beam_Selection/data/baseline_data/beam_output/beams_output_test.npz"
    test_data = LidarDataset3D(lidar_test_path, beam_test_path)

    model = LidarMarcus
    loss_fn = lambda y_true, y_pred: -tf.reduce_sum(tf.reduce_mean(y_true[y_pred>0] * tf.math.log(y_pred[y_pred>0]), axis=0))

    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy', dtype=None)
    top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy', dtype=None)
    optim = optimizers.Adam(lr=1e-3, epsilon=1e-8)

    scheduler = lambda epoch, lr: lr if epoch < 10 else lr/10.
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.compile(optimizer=optim, loss=loss_fn, metrics=[top1, top10])
    model.fit(training_data.lidar_data, training_data.beam_output, callbacks=callback, batch_size=16, epochs=20)
    model.evaluate(test_data.lidar_data, test_data.beam_output)


