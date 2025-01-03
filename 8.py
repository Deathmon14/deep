import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load and preprocess text
with open('your_text_file.txt', 'r') as file:
    text = file.read().lower()

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Generate input sequences
input_sequences = [seq[:i+1] for line in text.split('\n') for seq in [tokenizer.texts_to_sequences([line])[0]] for i in range(1, len(seq))]
max_seq_len = max(len(seq) for seq in input_sequences)
X = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')[:, :-1]
y = tf.keras.utils.to_categorical([seq[-1] for seq in input_sequences], total_words)

# Build LSTM model
model = Sequential([
    Embedding(total_words, 100, input_length=max_seq_len - 1),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(150),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X, y, epochs=100, verbose=1)

# Text generation function
def generate_text(seed_text, next_words, model, max_seq_len):
    for _ in range(next_words):
        token_list = pad_sequences([tokenizer.texts_to_sequences([seed_text])[0]], maxlen=max_seq_len - 1, padding='pre')
        seed_text += " " + tokenizer.index_word[np.argmax(model.predict(token_list, verbose=0))]
    return seed_text

# Generate example text
print(generate_text("your seed text", 10, model, max_seq_len))

# Plot training accuracy and loss
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.title('Model Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()
