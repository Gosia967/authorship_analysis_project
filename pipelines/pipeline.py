import typing as t
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics

from models import Model
from data_representation import Representation
from data_preparation import authors_abbr

class Pipeline:
    
    def __init__(
        self,
        corpus_train: np.array,
        corpus_test: np.array,
        classes_train: np.array,
        classes_test: np.array,
        class_names: list,
        representations: t.List[str],
        models: t.List[str],
        knn = 5,
        ann_args_dict: t.Dict[t.Tuple[str, str], list] = None
    ):
        self.corpus_train = corpus_train
        self.corpus_test = corpus_test
        self.classes_train = classes_train
        self.classes_test = classes_test
        self.class_names = class_names
        self.representations = representations
        self.models = models
        self.knn = knn
        self.ann_args_dict = ann_args_dict
        self.acc_list = []
        self.cm_list = []
        self.kappa_list =[]
        self.method_list = []  
        self.representations_for_expl = []
        self.models_for_expl = []    
        
    def run_pipeline(
        self,
        r: str,
        m: str,
    ):
        corpus_vect_test = np.copy(self.corpus_test)
        corpus_vect_train = np.copy(self.corpus_train)
        if r != 'empty':
            corpus_vect_train, corpus_vect_test = self._get_corpus_vects(r)
        ann_args = None
        if self.ann_args_dict:
            ann_args = self.ann_args_dict[(r,m)]
            ann_args.append(self.representations_for_expl[-1].tokenizer)
        model = Model(m, self.knn)
        model.fit(corpus_vect_train, self.classes_train, ann_args)
        classes_pred = model.predict(corpus_vect_test)
        acc = metrics.accuracy_score(self.classes_test, classes_pred)
        cm = metrics.confusion_matrix(self.classes_test, classes_pred, labels=self.class_names)
        kappa = metrics.cohen_kappa_score(self.classes_test, classes_pred)
        self.models_for_expl.append(model)
        return acc, cm, kappa
        
    def pipelines(self):
        for r in self.representations:
            for m in self.models:
                acc, cm, kappa = self.run_pipeline(r, m)
                self.acc_list.append(acc)
                self.cm_list.append(cm)
                self.kappa_list.append(kappa)
                if m == 'nn' and self.ann_args_dict[(r,m)][4] != 'basic':
                    self.method_list.append(self.ann_args_dict[(r,m)][4]+"+"+m)
                elif r == 'empty':
                    self.method_list.append(m)
                else:
                    self.method_list.append(r+"+"+m)
                
                
    def print_accuracy(self):
        accuracy_table = pd.DataFrame({"method": self.method_list,
                                       "accuracy": self.acc_list})
        print(accuracy_table)
        
    def accuracy_latex_format(self, path: str, latex_set_str: str):
        for i in range(len(self.acc_list)):
            train_corpus_size =  len(self.corpus_train) if self.method_list[i] != 'ng' else self.corpus_train[1]
            latex = f'{latex_set_str} & \
${len(self.corpus_test[0].split())}$ & \
${train_corpus_size}$ & \
${len(self.corpus_test)}$ & \
{self.method_list[i].upper()} & \
${round(self.acc_list[i], 3)}$ & \
${round(self.kappa_list[i], 3)}$\\\\'
            with open(path, 'a') as f:
                f.write(latex)
                f.write('\n')
            print(latex)
        
    def print_confussion_matrix(self):
        fig = plt.figure(figsize=(6*2, 6*len(self.cm_list)))
        for i in range(len(self.cm_list)):
            cm = self.cm_list[i]
            method = self.method_list[i]
            cm_pd = pd.DataFrame(cm)
            axex_description = authors_abbr(self.class_names)
            cm_pd.columns = axex_description
            cm_pd.index = axex_description
            sub = fig.add_subplot(len(self.cm_list), 1, i+1).set_title(method + ", acc="+ str(round(self.acc_list[i], 3)))
            #modyfikacja kodu https://www.kaggle.com/code/lvalencia/xgboost-fruit-classification
            cm_plot = sns.heatmap(cm_pd, annot=True, fmt='d', cmap='Greens')
            cm_plot.set_xlabel("Predicted values")
            cm_plot.set_ylabel("Actual values")
        fig.tight_layout()
        plt.show()
        
    def save_img(self, path: str):
        fig = plt.figure(figsize=(6*2, 6*len(self.cm_list)))
        for i in range(len(self.cm_list)):
            cm = self.cm_list[i]
            method = self.method_list[i].upper()
            cm_pd = pd.DataFrame(cm)
            axex_description = authors_abbr(self.class_names)
            cm_pd.columns = axex_description
            cm_pd.index = axex_description
            sub = fig.add_subplot(len(self.cm_list), 1, i+1).set_title(method + ", acc="+ str(round(self.acc_list[i], 3)))
            #modyfikacja kodu https://www.kaggle.com/code/lvalencia/xgboost-fruit-classification
            cm_plot = sns.heatmap(cm_pd, annot=True, fmt='d', cmap='Greens')
            cm_plot.set_xlabel("Predicted values")
            cm_plot.set_ylabel("Actual values")
        fig.tight_layout()
        plt.savefig(path)
          
    def _get_corpus_vects(self, r: str):
        repr = Representation(self.corpus_train, self.corpus_test, r)
        vects = repr.get_vectors() 
        self.representations_for_expl.append(repr)
        return vects  
    