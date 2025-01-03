import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
import keras_tuner as kt

# Define the model
def build_model(hp):
    units = hp.Int('units', 32, 128, step=32)
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    input_1 = Input(shape=(10,), name="input_1")
    x1 = layers.Dense(units, activation='relu')(input_1)
    x1 = layers.Dense(units, activation='relu')(x1)

    input_2 = Input(shape=(64, 64, 3), name="input_2")
    x2 = layers.Conv2D(32, (3, 3), activation='relu')(input_2)
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = layers.Flatten()(x2)

    combined = layers.concatenate([x1, x2])
    output_1 = layers.Dense(1, name="output_1")(combined)
    output_2 = layers.Dense(10, activation='softmax', name="output_2")(combined)

    model = Model(inputs=[input_1, input_2], outputs=[output_1, output_2])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={'output_1': 'mse', 'output_2': 'categorical_crossentropy'},
        metrics={'output_1': 'mae', 'output_2': 'accuracy'}
    )
    return model

# Set up tuner
tuner = kt.RandomSearch(
    build_model, objective='val_loss', max_trials=10, executions_per_trial=2,
    directory='my_dir', project_name='multi_input_multi_output_tuning'
)

# Generate random data
x1_data, x2_data = np.random.rand(1000, 10), np.random.rand(1000, 64, 64, 3)
y1_data, y2_data = np.random.rand(1000, 1), tf.keras.utils.to_categorical(np.random.randint(0, 10, (1000, 1)), 10)

# Split data
x1_train, x1_val = x1_data[:800], x1_data[800:]
x2_train, x2_val = x2_data[:800], x2_data[800:]
y1_train, y1_val = y1_data[:800], y1_data[800:]
y2_train, y2_val = y2_data[:800], y2_data[800:]

# Perform tuning
tuner.search([x1_train, x2_train], [y1_train, y2_train], validation_data=([x1_val, x2_val], [y1_val, y2_val]), epochs=5, batch_size=32)

# Train best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=5, batch_size=32)
