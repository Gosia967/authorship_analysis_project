import typing as t
import numpy as np
import kenlm
import os

from sklearn.base import BaseEstimator, ClassifierMixin

class NGramClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.path = os.path.join('..', 'arpa_files', 'arpa')

    def fit(self, X, y=None):
        self.authors_ord = sorted(list(set(y)))
        self.prep_descr = X[0]
        self.train_size_words = X[1]
        self.authors_models = self._get_language_models()
    
    def predict(self, X, y=None) -> t.List[str]:
        return [
            self.authors_ord[np.argmax(self._predictions_for_text(text))]
            for text in X
            ]
    
    def predict_proba(self, X, y=None):
        return [
            self._probabilities_for_text(text) for text in X
        ]
    
    def _get_language_models(self) -> t.List[kenlm.LanguageModel]:
        return [
            kenlm.LanguageModel(self._get_arpa_path(author))
            for author in self.authors_ord
        ]
        
    def _get_arpa_path(self, author: str):
        return os.path.join(self.path, author + '_' + self.prep_descr + str(self.train_size_words) + '.arpa')
    
    def _predictions_for_text(self, text: str) -> t.List[float]:
        return [model.score(text, bos = True, eos = True) for model in self.authors_models]
    
    def _probabilities_for_text(self, text: str) -> t.List[float]:
        pred = [10**x for x in self._predictions_for_text(text)]
        sum_pred = sum(pred)
        return [p/sum_pred for p in pred]
