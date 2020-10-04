
from keras.models import load_model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from bert_preprocessing import *

import os
import pickle
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model_dir = '/bert_models/'

model = load_model('weights_cpu.best.hdf5')
    
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
#    
MAX_SEQUENCE_LENGTH = model.input_shape[1]
global graph
graph = tf.get_default_graph()


# Prediction
def rate_toxic(text):
    text_clean = clean_text(text)
    text_split = text_clean.split(' ')
    
    # Tokenizer
    sequences = tokenizer.texts_to_sequences(text_split)
    sequences = [[item for sublist in sequences for item in sublist]]
    
    # Padding
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Prediction
    with graph.as_default():
        predict = model.predict(data).reshape(-1,1)
    return predict


