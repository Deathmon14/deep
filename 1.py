import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_image_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., None] / 255.0, x_test[..., None] / 255.0
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
    model.summary()
    return model

def build_tabular_model():
    data = {
        'size': [500, 1000, 1500, 2000, 2500],
        'bedrooms': [1, 2, 3, 4, 5],
        'age': [5, 10, 15, 20, 25],
        'price': [150000, 250000, 350000, 450000, 550000]
    }
    df = pd.DataFrame(data)
    X, y = df.drop('price', axis=1), df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.summary()
    return model

image_model = build_image_model()
tabular_model = build_tabular_model()
