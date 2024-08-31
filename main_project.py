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

def txt_clean(captions):
    table = str.maketrans('','',string.punctuation)
    for img, caps in captions.items():
        for i in range(len(caps)):
            descp = caps[i]
            descp = descp.split()
            #uppercase to lowercase
            descp = [word.lower() for word in descp]
            #remove punctuation from each token
            descp = [word.translate(table) for word in descp]
            #remove hanging 's and a
            descp = [word for word in descp if(len(word)>1)]
            #remove words containing numbers with them
            descp = [word for word in descp if(word.isalpha())]
            #converting back to string
            caps[i] = ' '.join(descp)
    
    return captions

def txt_vocab(descriptions):
  # To build vocab of all unique words
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()