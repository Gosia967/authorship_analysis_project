import numpy as np
import tensorflow as tf    
#tf.compat.v1.disable_v2_behavior() 
tf.keras.backend.set_learning_phase(True)
from tensorflow.keras.models import Sequential
from keras import layers
from keras import callbacks
from keras.utils import to_categorical
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# kod klasy częściowo został zainspirowany tutorialem https://realpython.com/python-keras-text-classification/
class NeuralNet():
    def __init__(self):
        self.model = Sequential()
    
    def fit(self):
        callback = callbacks.EarlyStopping(monitor='accuracy', patience=5)
        history = self.model.fit(
            self.X_train, 
            self.train_classes,
            epochs=self.epochs,
            verbose=False,
            validation_data=self.validation_data,
            #validation_split=0.2,
            batch_size=self.batch_size,
            callbacks=[callback])
        self.plot_history(history)
        
    def predict(self, X_test=None, y=None):
        X_test = self.adjust_test_set(X_test)
        predict_x = self.model.predict(X_test) 
        classes_idx = np.argmax(predict_x,axis=1)
        return np.array([self.classes_[idx] for idx in classes_idx]).T
        
    def predict_proba(self, X_test=None, y=None):
        X_test = self.adjust_test_set(X_test)
        predict_x = self.model.predict(X_test) 
        return np.array([
            [i/sum(p) for i in p] for p in predict_x
        ]).T
           
    def prepare_and_compile_model(self, X_train, train_classes, ann_args: list):
        self.X_train = X_train
        self.train_classes = train_classes
        self.select_k = ann_args[0]
        self.epochs = ann_args[1]
        self.batch_size = ann_args[2]
        self.layer_units = ann_args[3]
        self.layers_arch = ann_args[4] # 'basic' / 'lstm' / 'embed'
        self.tokenizer = ann_args[5]
        self.classes_ = list(set(self.train_classes))
              
        self.adjust_sets()      
        self.add_layers()
        self.compile()
        print(self.model.summary())
        
    def adjust_sets(self):
        if self.layers_arch == 'basic':
            self.select = SelectKBest(score_func=chi2, k=self.select_k)
            X_train_new = self.select.fit_transform(self.X_train, self.train_classes)
            self.X_train = X_train_new       
        self.train_classes = to_categorical(
            np.array([self.classes_.index(a) for a in self.train_classes]),
            len(self.classes_))
        train_val, classes_val, train_par, classes_par = self._prepare_validation_set(self.X_train, self.train_classes, len(self.classes_))
        self.X_train = train_par
        self.train_classes = classes_par       
        self.validation_data = (
            train_val,
            classes_val
        )
        
    def adjust_test_set(self, X_test):
        if self.layers_arch == 'basic':
            return self.select.transform(X_test)
        return X_test
               
    def add_layers(self):
        if self.layers_arch == 'basic':
            self._basic_layers()
        elif self.layers_arch == 'lstm':
            self._lstm_layers()
        elif self.layers_arch == 'embed_basic':
            self._embed_basic_layers()    
        elif self.layers_arch == 'embed_glove_lstm':
            self._embedding_glove_layers()      
        
    def compile(self):
        self.model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
               
    #funkcja pochodzi z tutoriala https://realpython.com/python-keras-text-classification/
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
        
    def _basic_layers(self):
        input_dim = self.X_train.shape[1]
        classes_num = len(self.classes_)
        self.model.add(layers.Dense(self.layer_units, input_dim=input_dim, activation='relu'))
        self.model.add(layers.Dense(classes_num, activation='softmax'))
        
    def _lstm_layers(self):
        input_length = self.X_train.shape[1]
        classes_num = len(self.classes_)
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model.add(layers.Embedding(vocab_size, 32, input_length=input_length))
        self.model.add(layers.LSTM(128))
        self.model.add(layers.Dense(classes_num, activation='softmax'))
    
    def _embed_basic_layers(self):
        input_length = self.X_train.shape[1]
        classes_num = len(self.classes_)
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model.add(layers.Embedding(vocab_size, 32, input_length=input_length))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(classes_num, activation='softmax'))
    
    def _embedding_glove_layers(self):
        embedding_dim = 100
        maxlen = 100
        classes_num = len(self.classes_)
        vocab_size = len(self.tokenizer.word_index) + 1  
        embedding_matrix = self._create_embedding_matrix(
            'embedding/glove_100_3_polish.txt',
            self.tokenizer.word_index, embedding_dim)
        self.model.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=False))
        self.model.add(layers.LSTM(128))
        self.model.add(layers.Dense(self.layer_units, activation='relu'))
        self.model.add(layers.Dense(classes_num, activation='softmax'))

    # funkcja pochodzi z tutoriala: https://realpython.com/python-keras-text-classification/
    def _create_embedding_matrix(self, filepath, word_index, embedding_dim):
        vocab_size = len(word_index) + 1  
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]
        return embedding_matrix
    
    def _prepare_validation_set(self, train, classes, classes_num):
        print(type(train))
        print(train.shape)
        if 'sparse' in str(type(train)):
            train = train.toarray()
        val_num = int(train.shape[0] / classes_num * 0.2)
        one_class_size = int(train.shape[0] / classes_num)
        train_val = np.array([sample for slist in [train[i*one_class_size:i*one_class_size+val_num] for i in range(classes_num)] for sample in slist])
        classes_val = np.array([sample for slist in [classes[i*one_class_size:i*one_class_size+val_num] for i in range(classes_num)] for sample in slist])
        train_par = np.array([sample for slist in [train[i*one_class_size+val_num:i*one_class_size+one_class_size] for i in range(classes_num)] for sample in slist])
        classes_par = np.array([sample for slist in [classes[i*one_class_size+val_num:i*one_class_size+one_class_size] for i in range(classes_num)] for sample in slist])
        print(type(train_val))
        print(train_val.shape, classes_val.shape)
        print(train_par.shape, classes_par.shape)
        return train_val, classes_val, train_par, classes_par
