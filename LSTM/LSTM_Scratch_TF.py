'''LSTM from scratch with TensorFlow'''

import tensorflow as tf
import keras
from keras import layers, Model 
import numpy as np


# LSTM Cell
class LSTMCell(layers.Layer):

    def __init__(self, units):
        super(LSTMCell, self).__init__()
        self.units  = units

    # inisialisasi bobot dan bias pada tiap gate (gerbang)
    def build(self, input_shape):
        input_dim = input_shape[-1] # mengambil dimensi input

        # inisialisasi bobot di forget gate
        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer='glorot_unifrorm', trainable=True)
        self.U_f = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
        self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        # inisialisasi bobot di input gate
        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.U_i = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        # inisialisasi bobot untuk kandidat di cell state
        self.W_c = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.U_c = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
        self.b_c = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        # inisialisasi bobot di output gate
        self.W_o = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.U_o = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    # proses forward
    def forward(self, inputs, states):
        h_prev, c_prev = states

        # operasi di forget gate
        f_t = tf.nn.sigmoid(tf.matmul(inputs, self.W_f) + tf.matmul(h_prev, self.U_f) + self.b_f)

        # operasi di input gate
        i_t = tf.nn.sigmoid(tf.matmul(inputs, self.W_i) + tf.matmul(h_prev, self.U_i) + self.b_i)

        # opreasi untuk kandidat cell state
        c_kand = tf.nn.tanh(tf.matmul(inputs, self.W_c) + tf.matmul(h_prev, self.U_c) + self.b_c)

        # operasi update cell state
        c_t = f_t * c_prev + i_t * c_kand

        # operasi di output gate
        o_t = tf.nn.sigmoid(tf.matmul(inputs, self.W_o) + tf.matmul(h_prev, self.U_o) + self.b_o)

        # hidden state yang baru
        h_t = o_t * tf.nn.tanh(c_t)

        return h_t, [h_t, c_t]
    
    # proses backward (backpropagation trough time)
    def backward(self, d_h_next, d_c_next):
        # gradien di output gate
        d_o_t = d_h_next * tf.nn.tanh(self.c_t)
        d_o_t *= self.o_t * (1 - self.o_t) # turunan sigmoid

        # gradien di cell state
        d_c_t = d_h_next * d_o_t * (1 - tf.square(tf.nn.tanh(self.d_c_next)))
        d_c_t += self.d_c_next

        # gradient di input gate
        d_i_t = d_c_t * self.c_kand
        d_i_t *= self.i_t * (1 - self.i_t) # turunan sigmoid

        # gradient di forget gate
        d_f_t = d_c_t * self.c_prev
        d_f_t *= self.f_t *  (1 - self.f_t) # turunan sigmoid

        # gradient di kandidat cell state
        d_c_kand = d_c_t * self.i_t 
        d_c_kand *= (1 - tf.square(self.c_kand)) # turunan tanh

        return d_h_next, d_c_t, d_i_t, d_f_t, d_c_kand 


# LSTM
class LSTM(layers.Layer):

    def __init__ (self, units):
        super(LSTM, self).__init__()
        self.units = units
        self.lstm = LSTMCell(units)

    # forward
    def forward(self, inputs):
        # unpack setiap elemen dari inputs shape
        batch_size, timesteps, _ = tf.unstack(tf.shape(inputs))

        # inisialisasi hidden state dan cell state dengan matriks 0
        h_t = tf.zeros((batch_size, self.units))
        c_t = tf.zeros((batch_size, self.units))

        # proses penhitungan tiap timesteps
        outputs = []
        for t in range(timesteps):
            output, [h_t, c_t] = self.lstm(inputs[:, t, :], [h_t, c_t])
            outputs.append(output)

        return tf.stack(outputs, axis=1)
    
    # backward - BPTT
    def backward(self, d_outputs):
        batch_size, timesteps, _ = tf.shape(d_outputs)
        d_h_next = tf.zeros((batch_size, self.units))
        d_c_next = tf.zeros((batch_size, self.units))

        for t in reversed(range(timesteps)):
            d_h_next, d_c_next = self.lstm.backward(d_h_next, d_c_next)
        return d_h_next, d_c_next
 