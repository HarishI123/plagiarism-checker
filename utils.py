import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
from keras.models import load_model
# text=load_model("/content/drive/MyDrive/humanAI.h5")


model = load_model('humanAI.h5') 
def preprocessdata(sentence):
    # convert sentence to sequence
    tokenizer = Tokenizer(num_words=6000)
    tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = pad_sequences(sequences, maxlen=300, padding="post", truncating="post")

    # get predictions for toxicity
    predictions = model.predict(padded_sequences)[0]
    predictions = random.randint(0, 1)
    
    if predictions > 0.5:
        return 1
    else:
        return 0


