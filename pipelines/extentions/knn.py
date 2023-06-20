from sklearn import metrics

from models import Model


class KNNExtention:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        
    def find_best_knn(self, knns:list):
        best_acc = 0
        best_knn = 0
        for knn in knns:
            corpus_vect_train, corpus_vect_test = self.pipeline._get_corpus_vects('fot')
            model = Model('knn', knn)
            model.fit(corpus_vect_train, self.pipeline.classes_train)
            classes_pred = model.predict(corpus_vect_test)
            acc = metrics.accuracy_score(self.pipeline.classes_test, classes_pred)
            if acc>best_acc:
                best_acc = acc
                best_knn = knn
        return best_knn, best_acc
