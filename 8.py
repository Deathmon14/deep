import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load and preprocess text
text = open('your_text_file.txt', 'r').read().lower()

# Tokenize and generate sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
seqs = [seq[:i+1] for line in text.split('\n') for seq in [tokenizer.texts_to_sequences([line])[0]] for i in range(1, len(seq))]
max_len = max(len(seq) for seq in seqs)
X = pad_sequences(seqs, maxlen=max_len, padding='pre')[:, :-1]
y = tf.keras.utils.to_categorical([seq[-1] for seq in seqs], total_words)

# Build and train model
model = Sequential([
    Embedding(total_words, 100, input_length=max_len - 1),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(150),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=100, verbose=1)

# Generate text
def gen_text(seed, words, model, max_len):
    for _ in range(words):
        seed += " " + tokenizer.index_word[np.argmax(model.predict(pad_sequences([tokenizer.texts_to_sequences([seed])[0]], maxlen=max_len - 1, padding='pre'), verbose=0))]
    return seed

# Generate and plot
print(gen_text("your seed text", 10, model, max_len))
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.show()
