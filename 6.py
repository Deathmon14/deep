import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import string

# Text data
text = """
In the beginning God created the heavens and the earth. Now the earth was formless and empty,
darkness was over the surface of the deep, and the Spirit of God was hovering over the waters. 
And God said, “Let there be light,” and there was light. God saw that the light was good,
and he separated the light from the darkness.
"""

# Clean and tokenize text
text = text.lower().translate(str.maketrans('', '', string.punctuation))
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
words = tokenizer.texts_to_sequences([text])[0]
sequences = [words[:i+1] for i in range(1, len(words))]
X = pad_sequences(sequences, padding='pre')[:, :-1]
y = tf.keras.utils.to_categorical(sequences, num_classes=total_words)[:, -1]

# Build and compile model
model = Sequential([
    Embedding(total_words, 100, input_length=X.shape[1]),
    LSTM(150),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=20, verbose=1)

# Generate text
def generate_text(seed, next_words):
    for _ in range(next_words):
        token_list = pad_sequences([tokenizer.texts_to_sequences([seed])[0]], maxlen=X.shape[1], padding='pre')
        seed += " " + tokenizer.index_word[np.argmax(model.predict(token_list))]
    return seed

# Generate and print new text
print(generate_text("God said", 10))
