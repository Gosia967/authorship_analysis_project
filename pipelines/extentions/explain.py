import numpy as np
import pandas as pd
import typing as t

from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from sklearn import tree
from lime import lime_text
import shap
import matplotlib.pyplot as plt
import graphviz 
import spacy

from data_preparation import authors_abbr
from data_preparation.preparation_utils import Preprocessing


class ArrayTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()


class Explain:
    
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        
    def classify(self, text: str, i: int):
        text_repr = self.pipeline.representations_for_expl[i].vectorizer.transform([text]).toarray()
        predicted_class = self.pipeline.models_for_expl[i].clf.predict(text_repr)
        predicted_proba = self.pipeline.models_for_expl[i].clf.predict_proba(text_repr)
        print(self.pipeline.models_for_expl[i].model_name, " ", self.pipeline.representations_for_expl[i].rep, ": ", predicted_class)
        print(self.pipeline.models_for_expl[i].clf.classes_)
        print(predicted_proba)
        print()
        return predicted_class
    
    def text_preprocessing(self, text: str, prep_func_names: t.List[str], pos: t.List[str] = []):
        nlp = spacy.load('pl_core_news_lg')
        nlp.max_length = 2500000
        preprocessing = Preprocessing(text, nlp)
        for pf_name in prep_func_names:
            pf = getattr(preprocessing, pf_name)
            if pf_name == 'POS_leave_only':
                pf(pos)
            else:
                pf()
        return preprocessing.text
        
    def lime(self, text: str, i: int, path: str, true_class: str):
        c_pipeline = make_pipeline(
            self.pipeline.representations_for_expl[i].vectorizer, 
            ArrayTransformer(),
            self.pipeline.models_for_expl[i].clf)
        explainer = lime_text.LimeTextExplainer(
            class_names=authors_abbr(self.pipeline.models_for_expl[i].clf.classes_))
        explanation = explainer.explain_instance(
            text, 
            c_pipeline.predict_proba,
            top_labels=len(self.pipeline.class_names), 
            num_samples=len(self.pipeline.corpus_train),
            #num_features=num_features
            )
        explanation.show_in_notebook()
        fig = explanation.as_pyplot_figure(self.pipeline.models_for_expl[i].clf.classes_.tolist().index(true_class))
        model_name = self.pipeline.models_for_expl[i].model_name
        repr_name = self.pipeline.representations_for_expl[i].rep
        path_jpg = path[:-4] + '_' + model_name + '_' + repr_name + '.jpg'
        path_html = path[:-4] + '_' + model_name + '_' + repr_name + '.html'
        explanation.save_to_file(path_html)
        fig.savefig(path_jpg)
        
    #kod do rozwoju, nieuwzględniony w pracy
    def shap(self, text: str, i: int):
        if self.pipeline.models_for_expl[i].model_name in ['mnb', 'gnb', 'cnb', 'svc', 'knn']:
            shap_values = shap.KernelExplainer(
                self.pipeline.models_for_expl[i].clf.predict,
                self.pipeline.representations_for_expl[i].X_test
            )
            shap.summary_plot(shap_values, self.pipeline.representations_for_expl[i].X_test)
        elif self.pipeline.models_for_expl[i].model_name in ['dtc', 'rfc']:
            explainer = shap.Explainer(self.pipeline.models_for_expl[i].clf)
            shap_values = explainer(self.pipeline.representations_for_expl[i].X_test)
            shap.summary_plot(shap_values, self.pipeline.representations_for_expl[i].X_test, plot_type="bar")
        elif self.pipeline.models_for_expl[i].model_name == 'nn':
            attrib_data = self.pipeline.models_for_expl[i].clf.adjust_test_set(self.pipeline.representations_for_expl[i].X_train[:200])
            explainer = shap.DeepExplainer(self.pipeline.models_for_expl[i].clf.model, attrib_data)
            #shap_values = explainer.shap_values(attrib_data)
            test_data = self.pipeline.models_for_expl[i].clf.adjust_test_set(self.pipeline.representations_for_expl[i].X_test[:10])
            shap_values = explainer.shap_values(attrib_data)
            print(shap_values)
            shap.summary_plot(shap_values)
            
    def explain_tree(self, i: int, path: str):
        if self.pipeline.models_for_expl[i].model_name == 'dtc':
            # modyfikacja kodu z tutoriala Scikit-learn https://scikit-learn.org/stable/modules/tree.html#tree
            plt.figure(figsize=(24,24)) 
            tree.plot_tree(self.pipeline.models_for_expl[i].clf, max_depth=6, fontsize=10)
            plt.show()
            
            dot_data = tree.export_graphviz(
                self.pipeline.models_for_expl[i].clf, 
                out_file=None,
                feature_names=self.pipeline.representations_for_expl[i].vectorizer.feature_names_,  
                class_names=self.pipeline.class_names,  
                filled=True, rounded=True,  
                special_characters=True)  
            graph = graphviz.Source(dot_data)  
            graph.render("DTC") 
            
            # r = tree.export_text(self.pipeline.models_for_expl[i].clf, feature_names=self.pipeline.representations_for_expl[i].vectorizer.feature_names_)
            # print(r)
            
            
            # modyfikacja kodu z tutoriala Scikit-learn: #https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
            importances = self.pipeline.models_for_expl[i].clf.feature_importances_
            std = np.std(importances, axis=0)
            dc_importances = pd.Series(importances, index=self.pipeline.representations_for_expl[i].vectorizer.feature_names_)
            fig, ax = plt.subplots()
            dc_importances.plot.bar(yerr=std, ax=ax)
            ax.set_title('Ważność cech')
            ax.set_ylabel('Istotność cechy')
            fig.tight_layout()
            fig.savefig(path)
            plt.show()
            
    def explain_forest(self, i: int, path: str):
        if self.pipeline.models_for_expl[i].model_name == 'rfc':
            # modyfikacja kodu z tutoriala Scikit-learn: #https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
            importances = self.pipeline.models_for_expl[i].clf.feature_importances_
            std = np.std([tree.feature_importances_ for tree in self.pipeline.models_for_expl[i].clf.estimators_], axis=0)
            forest_importances = pd.Series(importances, index=self.pipeline.representations_for_expl[i].vectorizer.feature_names_)
            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=std, ax=ax)
            ax.set_title('Ważność cech')
            ax.set_ylabel('Istotność cechy')
            fig.tight_layout()
            fig.savefig(path)
            plt.show()
