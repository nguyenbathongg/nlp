from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..core.interfaces import Vectorizer

class TextClassifier:
    """
    A text classifier that uses a Vectorizer to transform text into features
    and a LogisticRegression model for classification.
    """

    def __init__(self, vectorizer: Vectorizer):
        self._vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """
        Trains the classifier using the provided texts and labels.

        Args:
            texts: A list of text documents.
            labels: A list of corresponding integer labels.
        """
        # Transform texts into numerical features
        X = self._vectorizer.fit_transform(texts)
        
        # Initialize and train the Logistic Regression model
        # Use 'lbfgs' for multi-class problems (it's more stable)
        self._model = LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000)
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """
        Predicts labels for the given texts.

        Args:
            texts: A list of text documents to predict.

        Returns:
            A list of predicted integer labels.
        """
        if self._model is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")
        
        # Transform texts into numerical features
        X = self._vectorizer.transform(texts)
        
        # Make predictions
        return self._model.predict(X).tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Calculates and returns evaluation metrics.

        Args:
            y_true: The true labels.
            y_pred: The predicted labels.

        Returns:
            A dictionary containing accuracy, precision, recall, and f1_score.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        return metrics