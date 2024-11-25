import tensorflow as tf
from tensorflow.keras.datasets import mnist, imdb

# MNIST
print("Training MNIST Model")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

mnist_model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

mnist_model.fit(x_train, y_train, epochs=5)
mnist_test_loss, mnist_test_acc = mnist_model.evaluate(x_test, y_test)
print(f"MNIST Test Accuracy: {mnist_test_acc:.4f}\n")

# IMDB
print("Training IMDB Model")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=256)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=256)

imdb_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

imdb_model.compile(optimizer='adam', 
                   loss='binary_crossentropy', 
                   metrics=['accuracy'])

imdb_model.fit(x_train, y_train, epochs=5)
imdb_test_loss, imdb_test_acc = imdb_model.evaluate(x_test, y_test)
print(f"IMDB Test Accuracy: {imdb_test_acc:.4f}")