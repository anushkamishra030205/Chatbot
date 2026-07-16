import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')  # newer NLTK versions need this

lemmatizer = WordNetLemmatizer()

# ── FIX 1: Use raw string (r"") for Windows path to avoid escape issues ──
intents = json.loads(open(r'C:\Users\ASUS\OneDrive\Documents\Desktop\chatbot (2)\intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# ── FIX 2: Lemmatize + lowercase together (was missing .lower()) ──
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words,   open('words.pkl',   'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1 if word in wordPatterns else 0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# ── FIX 3: Model build using Input layer (input_shape in Dense is deprecated) ──
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(trainX[0]),)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# ── FIX 4: model.save() doesn't take hist as argument ──
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# ── FIX 5: Save as .keras format (h5 is legacy, causes warning in TF2.x+) ──
model.save('chatbot_model.keras')

print('Done! Model saved as chatbot_model.keras')
