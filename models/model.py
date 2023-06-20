from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.backend import clear_session
from .n_gram import NGramClassifier
from .neural_net import NeuralNet

class Model:
    #nbm, nbg, nbc, dtc, rfc, svc, svc_, knn, ng, nn
    def __init__(self, model_name: str, knn: int):
        self.model_name = model_name
        self.clf = None
        if model_name == 'mnb':
            self.clf = MultinomialNB()
        elif model_name == 'gnb':
            self.clf = GaussianNB()
        elif model_name == 'cnb':
            self.clf = ComplementNB()
        elif model_name == 'dtc':
            self.clf = DecisionTreeClassifier()
        elif model_name == 'rfc':
            self.clf = RandomForestClassifier()
        elif model_name == 'svc':
            self.clf = LinearSVC()
        elif model_name == 'svc_':
            self.clf = SVC(kernel='linear', probability=True)
        elif model_name == 'knn':
            self.clf = KNeighborsClassifier(knn)
        elif model_name == 'ng':
            self.clf = NGramClassifier()
        elif model_name == 'nn':
            self.clf = NeuralNet()
        
    def fit(self, corpus_vect_train, classes_train, ann_args: list = None):
        if self.model_name == 'nn':
            self.clf.prepare_and_compile_model(corpus_vect_train, classes_train, ann_args)
            self.clf.fit()
            clear_session()
        else:
            self.clf.fit(corpus_vect_train, classes_train)
        
    def predict(self, corpus_vect_test):
        classes_pred = self.clf.predict(corpus_vect_test)
        return classes_pred