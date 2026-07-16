import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

# ── FIX 1: Use raw string for Windows path ──
intents = json.loads(open(r'C:\Users\ASUS\OneDrive\Documents\Desktop\chatbot (2)\intents.json').read())

words   = pickle.load(open('words.pkl',  'rb'))
classes = pickle.load(open('classes.pkl','rb'))

# ── FIX 2: Load .keras file (updated from legacy .h5) ──
# If you saved as .h5 previously, change back to 'chatbot_model.h5'
model = load_model('chatbot_model.keras')


# ── FIX 3: Added .lower() in lemmatize (consistent with training code) ──
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]  # ── FIX 4: verbose=0 (hides prediction logs) ──

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({
            'intent':      classes[r[0]],
            'probability': str(r[1])
        })
    return return_list


# ── FIX 5: Handle empty intents_list (no match found) ──
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Please try again!"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result

    return "Sorry, I couldn't find a response!"  # ── FIX 6: Fallback if tag not matched ──


print("GO! Bot is running! (type 'quit' to exit)\n")

# ── FIX 7: Added quit option to exit loop cleanly ──
while True:
    message = input("You: ").strip()

    if not message:               # ── FIX 8: Skip empty input ──
        continue

    if message.lower() == 'quit':
        print("Bye! Bot stopped.")
        break

    ints = predict_class(message)
    res  = get_response(ints, intents)
    print(f"Bot: {res}\n")