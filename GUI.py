import tkinter as tk
from tkinter import scrolledtext, END
import json
import pickle
import numpy as np
from keras.models import load_model

# Load the chatbot model
model = load_model('./chatbot_model.h5')

# Load the preprocessed data
words = pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))
intents = json.loads(open('./intents.json').read())

# Create the GUI window
window = tk.Tk()
window.title("Chatbot")
window.geometry("400x500")

# Create the chat window
chat_window = scrolledtext.ScrolledText(window, wrap=tk.WORD)
chat_window.pack(fill=tk.BOTH, expand=True)
chat_window.configure(state="disabled")

# Create the input box
input_box = tk.Entry(window, width=40)
input_box.pack(pady=10)

# Function to handle user input and display chatbot response
def send_message():
    user_message = input_box.get()
    input_box.delete(0, tk.END)
    chat_window.configure(state="normal")
    chat_window.insert(tk.END, "You: " + user_message + "\n")
    chat_window.configure(state="disabled")
    
    ints = predict_class(user_message, model)
    response = get_response(ints, intents)
    
    chat_window.configure(state="normal")
    chat_window.insert(tk.END, "Bot: " + response + "\n")
    chat_window.configure(state="disabled")
    chat_window.see(tk.END)

# Function to predict the intent of user input
def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get the chatbot response based on predicted intent
def get_response(ints, intents):
    tag = ints[0]['intent']
    for intent in intents:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            break
    return response

# Function to preprocess user input
def bow(sentence, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(sentence)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Bind the Enter key to send a message
window.bind('<Return>', lambda event: send_message())

# Create the Send button
send_button = tk.Button(window, text="Send", command=send_message)
send_button.pack()

# Run the GUI main loop
window.mainloop()