"""Clasificadores auxiliares compatibles con scikit-learn."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC


class SVMSoftmax(ClassifierMixin, BaseEstimator):
    """SVM lineal que entrega probabilidades con un softmax sobre su función de decisión.

    ``CalibratedClassifierCV`` no se puede usar porque hay clases con 1 o 2
    ejemplos. El softmax no está calibrado, pero conserva el orden de las clases,
    que es lo que importa para el top 1 (macro-F1) y el top 3 (MAP@3).
    """

    def __init__(self, C: float = 0.1, temperatura: float = 1.0, max_iter: int = 5000):
        self.C = C
        self.temperatura = temperatura
        self.max_iter = max_iter

    def fit(self, X, y):
        self.svm_ = LinearSVC(C=self.C, max_iter=self.max_iter).fit(X, y)
        self.classes_ = self.svm_.classes_
        return self

    def decision_function(self, X):
        return self.svm_.decision_function(X)

    def predict_proba(self, X):
        z = self.decision_function(X) / self.temperatura
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
