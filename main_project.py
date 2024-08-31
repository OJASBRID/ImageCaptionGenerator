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
from tqdm.notebook import tqdm as tqdm 



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
        tokens = line.split(",")
        
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
    
dataset_text = r"D:\Dataset\captions.txt"
dataset_images = r"D:\Dataset\Images"

filename = dataset_text
descriptions = img_capt(filename)
print("Length of descriptions =" ,len(descriptions))
#cleaning the descriptions
clean_descriptions = txt_clean(descriptions)
#to build vocabulary
vocabulary = txt_vocab(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))
#saving all descriptions in one file
save_descriptions(clean_descriptions, "descriptions.txt")

model = Xception( include_top=False, pooling='avg' )

def extract_features(directory):
    model = Xception( include_top=False, pooling='avg' )
    features = dict() 

    for pic in os.listdir(directory):
        file = directory + "/" + pic
        image = Image.open(file)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis = 0)
        image = image / 127.5
        image = image - 1.0
        feature = model.predict(image)
        features[pic] = feature
    
    return features

features = extract_features(dataset_images)

len(features)
dump(features, open("features.p", "wb"))

features  = load(open("features.p", "rb"))

print(features)

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos

filename = r"D:\Dataset\image_names.txt"
# train = loading_data(filename)
train_imgs = load_photos(filename)

print(len(train_imgs))

doc = load_doc("descriptions.txt")
print(doc)

print(train_imgs)

def load_clean_descriptions(filename, photos):
    #loading clean_descriptions
    doc = load_doc(filename)
    descriptions = dict()

    for line in doc.split('\n'):
        words = line.split()
        if len(words) < 1:
            continue

        image_id, image_caption = words[0], words[1:]

        image_id_ext = image_id + ".jpg"

        if image_id_ext in photos:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            cap_gem = 'startseq ' + ' '.join(image_caption) + ' endseq' 
            descriptions[image_id].append(cap_gem)

    return descriptions


train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)

def load_features(photos):
    #loading all features
    all_features = load(open("features.p","rb"))
    #selecting only needed features
    features = {k:all_features[k] for k in photos}
    return features

train_features = load_features(train_imgs)

def dict_to_list(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

from keras._tf_keras.keras.preprocessing.text import Tokenizer
def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
vocab_size 