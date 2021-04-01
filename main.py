import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import string
import keras
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
plt.style.use('ggplot')
count_edu = 0
count_game = 0
count_an = 0
count_sec=0
count_fin = 0
count_crm = 0
count_coms = 0
categories = ['Education and Learning' , "Game Development" , "Analytics and Intelligence" , "Security" , "Accounting and Finance" , "CRM" , "Communications"]
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

num_classes = 27
snowball = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()
pd.set_option('display.max_columns', None)
df = pd.read_csv('Repositories.csv')
df.ReadMe = df.ReadMe.astype(str)
df.head()
df.to_numpy()
logging.info("Transforming dataset values...")
for x in range(0, df.shape[0]):
    df['ReadMe'].loc[x] = df['ReadMe'].loc[x].lower()
for x in range(0, df.shape[0]):
    df['ReadMe'].loc[x] = df['ReadMe'].loc[x].translate(str.maketrans('', '', string.punctuation))
    df['ReadMe'].loc[x] = df['ReadMe'].loc[x].replace(r'\W', "")
    df['ReadMe'].loc[x] = df['ReadMe'].loc[x].replace("\n", "")

#COUNTING EACH CATEGORIES APPEARENCE IN DATASET
for x in range (0, df.shape[0]):
    if df['Category'].loc[x] == "Education":
        count_edu = count_edu+1
    elif df['Category'].loc[x] == "Game Development":
        count_game= count_game+1
    elif df['Category'].loc[x] == "Analytics":
        count_an = count_an+1
    elif df['Category'].loc[x] == "Security":
        count_sec = count_sec+1
    elif df['Category'].loc[x] == "Finance":
        count_fin = count_fin +1
    elif df['Category'].loc[x] == "CRM":
        count_crm = count_crm + 1
    elif df['Category'].loc[x] == "Communications":
        count_coms = count_coms + 1
    else:
        print("Marie giati uparxei to category")

counted = [count_edu,count_game , count_an , count_sec, count_fin , count_crm , count_coms]
print(counted)
"""
fig = plt.figure(figsize=(10, 10))
for x in range(0,6):
    plt.bar(categories[x],counted[x])                   
plt.ylabel('Number of instances in dataset')
plt.xlabel('Categories in dataset')
plt.legend(labels=['Education', 'Game Development' ,'Analytics', 'Security', 'Finance' , 'CRM' , 'Communications'])
plt.show()
"""
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df['Category'])
labels = multilabel_binarizer.classes
maxlen = 500
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df['ReadMe'])


def get_features(text_series):
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences,maxlen =maxlen)

def prediction_to_label(prediction):
    tag_prob = [(labels[i], prob) for i , prob in enumerate(prediction.tolist())]
    return dict(sorted(tag_prob, key=lambda kv:kv[1] , reverse=True))

x = get_features(df['ReadMe'])
y = multilabel_binarizer.transform(df['Category'])
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9000)

filter_length = 500

model = Sequential()
model.add(Embedding(max_words, 20, input_length=maxlen))
model.add(Dropout(0.1))
model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)
]
logging.info("Keras model training started...")
history = model.fit(x_train, y_train,epochs=20, batch_size=32, validation_split=0.1, callbacks=callbacks)
logging.info("Evaluating the models accuracy...")
cnn_model = keras.models.load_model('model-conv1d.h5')
metrics = cnn_model.evaluate(x_test, y_test)
print("{}: {}".format(model.metrics_names[0], metrics[0]))
print("{}: {}".format(model.metrics_names[1], metrics[1]))
plot_history(history)

test_features = input("Enter description:")
#test_features = tokenizer.fit_on_texts(test_features)
model = tf.keras.models.load_model("model-conv1d.h5")
prediction = model.predict_classes(tf.convert_to_tensor(np.array(tokenizer.texts_to_sequences(test_features))),dtype=tf.float32)
classes = ["Education","Game Development","Analytics","Security","Finance","CRM","Lul" ]
#print(classes[int(prediction[0][0])])
print(prediction)


