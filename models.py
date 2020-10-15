from tensorflow.keras import layers, models, initializers

Lidar2D = models.Sequential([
    layers.Input(shape=(1, 20, 200)),
    layers.Conv2D(10, 3, 1, padding='same', kernel_initializer=initializers.HeUniform, data_format='channels_first'),
    layers.BatchNormalization(axis=1),
    layers.PReLU(),
    layers.Conv2D(10, 3, 1, padding='same', kernel_initializer=initializers.HeUniform, data_format='channels_first'),
    layers.BatchNormalization(axis=1),
    layers.PReLU(),
    layers.Conv2D(10, 3, 2, padding='same', kernel_initializer=initializers.HeUniform, data_format='channels_first'),
    layers.BatchNormalization(axis=1),
    layers.PReLU(),
    layers.Conv2D(10, 3, 1, padding='same', kernel_initializer=initializers.HeUniform, data_format='channels_first'),
    layers.BatchNormalization(axis=1),
    layers.PReLU(),
    layers.Conv2D(10, 3, 2, padding='same', kernel_initializer=initializers.HeUniform, data_format='channels_first'),
    layers.BatchNormalization(axis=1),
    layers.PReLU(),
    layers.Conv2D(10, 3, (1, 2), padding='same', kernel_initializer=initializers.HeUniform, data_format='channels_first'),
    layers.BatchNormalization(axis=1),
    layers.PReLU(),
    layers.Flatten(),
    layers.Dense(256),
    layers.ReLU(),
    layers.Dropout(0.7),
    layers.Dense(256),
    layers.Softmax()
])
