import numpy as np
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class NeuralNet():
    def __init__(self):
        self.model = Sequential()
    
    def fit(self):
        history = self.model.fit(
            self.X_train, 
            self.train_classes,
            epochs=self.epochs,
            verbose=False,
            validation_data=self.validation_data,batch_size=self.batch_size)
        self.plot_history(history)
        
    def predict(self, X_test=None, y=None):
        predict_x = self.model.predict(self.validation_data[0]) 
        classes_idx = np.argmax(predict_x,axis=1)
        return [self.authors[idx] for idx in classes_idx]
        
    def predict_proba(self, X_test=None, y=None):
        predict_x = self.model.predict(self.validation_data[0]) 
        return [
            [i/sum(p) for i in p] for p in predict_x
        ]
           
    def prepare_and_compile_model(self, X_train, train_classes, ann_args: list):
        self.X_train = X_train
        self.train_classes = train_classes
        self.select_k = ann_args[0]
        self.epochs = ann_args[1]
        self.validation_data = ann_args[2]
        self.batch_size = ann_args[3]
        self.layer_units = ann_args[4]
        self.layers_arch = ann_args[5] # 'basic' / 'lstm' / 'embed'
        self.tokenizer = ann_args[6]
        self.authors = list(set(self.train_classes))
        
        self.adjust_sets()      
        self.add_layers()
        self.compile()
        #self.model.build(input_shape = (100, 100, 100))
        print(self.model.summary())
        
    def adjust_sets(self):
        if self.layers_arch == 'basic' or self.layers_arch == 'lstm':
            self.select = SelectKBest(score_func=chi2, k=self.select_k)
            X_train_new = self.select.fit_transform(self.X_train, self.train_classes)
            self.validation_data = (
                self.select.transform(self.validation_data[0]),
                self.validation_data[1])
            self.X_train = X_train_new       
        self.train_classes = to_categorical(
            np.array([self.authors.index(a) for a in self.train_classes]),
            len(self.authors))       
        self.validation_data = (
            self.validation_data[0],
            to_categorical(
                np.array([self.authors.index(a) for a in self.validation_data[1]]),
                len(self.authors))
        )
               
    def add_layers(self):
        if self.layers_arch == 'basic':
            self._basic_layers()
        elif self.layers_arch == 'lstm':
            self._lstm_layers() 
        elif self.layers_arch == 'embed':
            self._embedding_layers()      
        
    def compile(self):
        self.model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
               
    def plot_history(self, history):
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
        
    def _basic_layers(self): #mlp
        input_dim = self.X_train.shape[1]
        classes_num = len(self.authors)
        self.model.add(layers.Dense(self.epochs, input_dim=input_dim, activation='relu'))
        self.model.add(layers.Dense(classes_num, activation='softmax'))
        
    def _lstm_layers(self):
        input_dim = self.X_train.shape[1]
        classes_num = len(self.authors)
        #vocab_size = len(self.tokenizer.word_index) + 1
        #vocab_size = self.tokenizer.vocab_size
        maxlen = 100
        self.model.add(layers.Embedding(800,16, input_length=input_dim))
        #self.model.add(layers.Dense(self.epochs, input_dim=input_dim, activation='relu'))
        #batch_input_shape=(batch_size, look_back, 1)
        #stateful=True,
        #input_dim=self.X_train.shape[1], 
        self.model.add(layers.CuDNNLSTM(32, return_sequences=True)) 
        self.model.add(layers.CuDNNLSTM(32))
        self.model.add(layers.Dense(self.epochs, input_dim=input_dim, activation='relu'))
        self.model.add(layers.Dense(classes_num, activation='softmax'))
        
    def _embedding_layers(self):
        embedding_dim = 100
        maxlen = 100
        classes_num = len(self.authors)
        vocab_size = len(self.tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        embedding_matrix = self._create_embedding_matrix(
            'embedding/glove_100_3_polish.txt',
            self.tokenizer.word_index, embedding_dim)
        self.model.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))
        self.model.add(layers.GlobalMaxPool1D())
        self.model.add(layers.Dense(self.epochs, activation='relu'))
        self.model.add(layers.Dense(classes_num, activation='softmax'))

    def _create_embedding_matrix(self, filepath, word_index, embedding_dim):
        vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        with open(filepath) as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]
        return embedding_matrix
