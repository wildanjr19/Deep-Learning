'''RNN from sctratch with NumPy'''

import numpy as np

class RNN():

    # inisialisasi input_size, hidden_size, dan output_size
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        '''
        input_size = jumlah fitur input
        hiddden_size = jumlah unit tersembunyi
        output_size = jumlah fitur outpur
        learning_rate = laju pembelajaran untuk pembaruan bobot
        '''
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # inisialisasi bobot
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01 # bobot dari input ke hidden (x ke h)
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # bobot dari hidden ke hidden (h ke h)
        self.Why = np.random.randn(output_size, hidden_size) * 0.01 # bobot dari hidden ke output ( h ke y)

        # inisialisasi bias
        self.bh = np.zeros((hidden_size, 1)) # bias untuk hidden
        self.by = np.zeros((output_size, 1)) # bias untuk output

    # fungsi forward_pass
    def forward_pass(self, inputs):
        
        h_t = np.zeros((self.hidden_size, 1)) 
        # h_t : vektor kolom
        # h_t akan menyimpan hidden state dari timestep sebelumnya

        self.hidden_states = [] # list kosong untuk menyimpan hidden state setiap timestep

        for x in inputs:
            x = x.reshape(-1, 1) # vektor kolom
            h_t = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_t) + self.bh)
            # persamaan ini menghitung hidden state baru (h_t) berdasarkan input saat ini(x)
            # dan hidden state sebelumnya (h_t)
            self.hidden_states.append(h_t)

        # hitung output
        y_t = np.dot(self.Why, h_t) + self.by
        return y_t, h_t # mengembalikan output dan hidden state terakhir

    # forward pass (menggunakan fungsi forward_pass untuk menghitung y_t dan h_t)
    def forward(self, inputs):
        return self.forward_pass(inputs)
    
    # proses backward (backpropagation : memperbarui bobot)
    def backward(self, inputs, targets, learning_rate=0.001):
    
        # mendapatkan output
        y_t, h_t = self.forward_pass(inputs)

        # mendapatkan loss
        loss = np.square(y_t - targets).sum()

        # Gradien BPPT
        dl_dWhy = 2 * (y_t - targets) @ h_t.T # gradien Why
        dl_dby = 2 * (y_t - targets) # gradien by

        # inisialisasi gradien dengan nol
        dWhh = np.zeros_like(self.Whh)
        dWxh = np.zeros_like(self.Wxh)
        dbh = np.zeros_like(self.bh)
        dh_next = np.zeros_like(h_t)

        # proses BPPT - setiap timestep
        for t in reversed(range(len(inputs))):
            dh = (1 - self.hidden_states[t] ** 2) * dh_next # turunan tanh dikali dengan hidden state selanjutnya
            dWhh += dh @ self.hidden_states[t-1].T # perbarui gradien Whh
            dWxh += dh @ inputs[t].reshape(1, -1) # perbarui gradien Wxh
            dbh += dh # perbarui gradien bias h
            dh_next = np.dot(self.Whh.T, dh) # propagasi balik hidden state

        # pembaruan bobot
        self.Whh -= learning_rate * dWhh
        self.Wxh -= learning_rate * dWxh
        self.Why -= learning_rate * dl_dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dl_dby

        return loss
    
    # train
    def train(self, data, targets, epochs):
        for epoch in range(epochs):
            loss = self.backward(data, targets)
            if epoch % 10 == 0:
                print(f'Epoch : {epoch}, Loss : {loss}') 

'''CONTOH PENGGUNAAN'''
np.random.seed(42)
rnn = RNN(input_size=3, hidden_size=5, output_size=1, learning_rate=0.001)

inputs = [np.random.randn(3) for _ in range(4)]
targets = np.random.randn(1, 1)

rnn.train(inputs, targets, epochs=100)

