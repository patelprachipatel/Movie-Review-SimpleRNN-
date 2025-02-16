### Step 1: Import the libraries and load the model
import pandas as pd
import numpy as np 
import re

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences


# load IMDB dataset index
word_index =  imdb.get_word_index()
reverse_word_index = {value : key for key, value in word_index.items()}


# Load the pretrained model with'Relu' activation
model=load_model('simple_rnn_imdb.h5')

### Step 2: Helper Function
## Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

## Function to preprocess user input
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    encoded_review = [min(i, 9999) for i in encoded_review]  
    padded_word = pad_sequences([encoded_review], maxlen=500)
    return padded_word


### Step 3: Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    print(prediction)
    return sentiment, prediction[0][0]


# streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as Positive or Negative.')


# User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    # preprocessed_input = preprocess_text(user_input)
    # prediction = model.predict(preprocessed_input)
    # sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    sentiment,score = predict_sentiment(user_input)
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write('Please enter a movie review')
    

