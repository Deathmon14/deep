import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import string

# --- Step 1: Text Preprocessing ---
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

# --- Step 2: Create Sequences ---
sequences = []
for i in range(1, len(tokenizer.texts_to_sequences([text])[0])):
    n_gram_sequence = tokenizer.texts_to_sequences([text])[0][:i+1]
    sequences.append(n_gram_sequence)

# Pad sequences
max_seq_len = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')

# Split into X (inputs) and y (targets)
X, y = sequences[:, :-1], sequences[:, -1]

# One-hot encode y
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# --- Step 3: Build the Model ---
model = Sequential([
    Embedding(total_words, 100, input_length=X.shape[1]),
    LSTM(150),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# --- Step 4: Train the Model ---
history = model.fit(X, y, epochs=20, verbose=1)

# --- Step 5: Generate Text ---
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=X.shape[1], padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)[0]
        output_word = tokenizer.index_word.get(predicted, "")
        if output_word == "":
            break
        seed_text += " " + output_word
    return seed_text

# Example usage
print("Generated Text:\n", generate_text("God said", 10))