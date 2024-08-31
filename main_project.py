import numpy as np
from PIL import Image
import os
import string
from pickle import dump
from pickle import load
from keras._tf_keras.keras.applications.xception import Xception #to get pre-trained model Xception
from keras._tf_keras.keras.applications.xception import preprocess_input
from keras._tf_keras.keras.preprocessing.image import load_img
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.preprocessing.text import Tokenizer #for text tokenization
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.layers import Input, Dense#Keras to build our CNN and LSTM
from keras._tf_keras.keras.layers import LSTM, Embedding, Dropout


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def img_capt(filename):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split("\t")
        
        # take the first token as image id, the rest as description
        image_id, image_desc = tokens[0], tokens[1:]
        
        # extract filename from image id
        image_id = image_id.split('.')[0]
        
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        if image_id not in descriptions:
            descriptions[image_id] = list()
        descriptions[image_id].append(image_desc)
    
    return descriptions