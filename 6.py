import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

text = "This is a simple example of language modeling using RNN."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences([text])[0]

input_sequences = [sequences[:i+1] for i in range(1, len(sequences))]
input_sequences = pad_sequences(input_sequences, padding='pre')
X, y = input_sequences[:, :-1], to_categorical(input_sequences[:, -1], total_words)

model = Sequential([
    Embedding(total_words, 10, input_length=X.shape[1]),
    SimpleRNN(50),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=50, verbose=0)

seed_text = "This is a"
token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences([token_list], maxlen=X.shape[1], padding='pre')
predicted_word = tokenizer.index_word[np.argmax(model.predict(token_list))]
print(f"Predicted word: {predicted_word}")
