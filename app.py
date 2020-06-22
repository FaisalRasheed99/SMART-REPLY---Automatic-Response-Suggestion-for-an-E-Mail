import os
import pandas as pd
import numpy as np
import nltk
import string
import re
from flask import Flask,request,jsonify,render_template
import pickle
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.enable_eager_execution()
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Dense,concatenate,Activation,Dropout,Input
from tensorflow.keras.models import Model
import keras
import keras.utils
from tensorflow.keras import utils as np_utils
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, BatchNormalization, Dense, concatenate
from tensorflow.keras import regularizers
from keras.utils import plot_model

app = Flask(__name__)

data=pd.read_csv("data.csv")

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
       message = request.form['review_text']
       print(message)
       question = message
       print('TYpe:',type(question))
       question = re.sub(r"won't", "will not", question)
       question = re.sub(r"can\'t", "can not", question)
       question = re.sub(r"n\'t", " not", question)
       question = re.sub(r"\'re", " are", question)
       question = re.sub(r"\'s", " is", question)
       question = re.sub(r"\'d", " would", question)
       question = re.sub(r"\'ll", " will", question)
       question = re.sub(r"\'t", " not", question)
       question = re.sub(r"\'ve", " have", question)
       question = re.sub(r"\'m", " am", question)
       question = question.replace('\\r', ' ')
       question = question.replace('\\"', ' ')
       question = question.replace('\\n', ' ')
       question = re.sub('[^A-Za-z0-9]+', ' ', question)
       question.lower().strip()
       question=question.split(" ")
       with open('tokenizer.pickle', 'rb') as h:
            t = pickle.load(h)
       with open('word2id.pickle', 'rb') as i:
            word2id=pickle.load(i)
       with open('id2word.pickle', 'rb') as j:
            id2word=pickle.load(j)
       max_ques_length=10
       max_ans_length=5
       vocab_size = len(t.word_index) + 1
       embedding_dim = 300
       units=1024
       BATCH_SIZE = 64
       question = t.texts_to_sequences([question])
       question = pad_sequences(question, maxlen=max_ques_length, padding='post')
       with open('embedding_matrix.pickle', 'rb') as k:
            embedding_matrix = pickle.load(k)
       def gru(units):
           return tf.keras.layers.GRU(units,return_sequences=True,return_state=True,recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')
       class Encoder(tf.keras.Model):
             def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
                 super(Encoder, self).__init__()
                 self.batch_sz = batch_sz
                 self.enc_units = enc_units
                 self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_ques_length, weights=[embedding_matrix], trainable=False)
                 self.gru = gru(self.enc_units)
             def call(self, x, hidden):
                  x = self.embedding(x)
                  output, state = self.gru(x, initial_state = hidden)        
                  return output, state
             def initialize_hidden_state(self):
                 return tf.zeros((self.batch_sz, self.enc_units))
       class Decoder(tf.keras.Model):
              def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
                    super(Decoder, self).__init__()
                    self.batch_sz = batch_sz
                    self.dec_units = dec_units
                    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_ans_length, weights=[embedding_matrix], trainable=False)
                    self.gru = gru(self.dec_units)
                    self.fc = tf.keras.layers.Dense(vocab_size)
                    self.drop_out = tf.keras.layers.Dropout(rate=0.3)
                    self.W1 = tf.keras.layers.Dense(self.dec_units)
                    self.W2 = tf.keras.layers.Dense(self.dec_units)
                    self.V = tf.keras.layers.Dense(1)
              def call(self, x, hidden, enc_output):
                   hidden_with_time_axis = tf.expand_dims(hidden, 1)
                   score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
                   attention_weights = tf.nn.softmax(score, axis=1)
                   context_vector = attention_weights * enc_output
                   context_vector = tf.reduce_sum(context_vector, axis=1)
                   x = self.embedding(x)
                   x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
                   output, state = self.gru(x)
                   output = tf.reshape(output, (-1, output.shape[2]))
                   x = self.fc(output)
                   return x, state, attention_weights
              def initialize_hidden_state(self):
                  return tf.zeros((self.batch_sz, self.dec_units))
       encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)
       decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)
       encoder.set_weights("encoder_weights.h5")
       decoder.set_weights("decoder_weights.h5")
       actual_sent = ''
       attention_plot = np.zeros((max_ans_length,max_ques_length ))
       sentence = ''
       question = tf.convert_to_tensor(question)
       result = ''
       hidden = [tf.zeros((1, units))]
       enc_out, enc_hidden = encoder(question, hidden)
       dec_hidden = enc_hidden
       dec_input = tf.expand_dims([word2id['<sos>']], 0)
       for t in range(max_ans_length-2): 
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
            attention_weights = tf.reshape(attention_weights, (-1, ))
            #attention_plot[t] = attention_weights.numpy()
            print(predictions[0])
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += id2word[predicted_id] + ' '
            if id2word[predicted_id] == '<eos>':
                return result 
            dec_input = tf.expand_dims([predicted_id], 0)
       print(result)
       return result
       #return render_template('index.html', prediction_text='Predicted E-mail response : $ {}'.format(result))
    

if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0',port=8080)
