"""
TF-IDF Vectorizer implementation for improved text representation.
"""

from typing import Dict, List
import math
from collections import Counter
from ..core.interfaces import Tokenizer, Vectorizer


class TfidfVectorizer(Vectorizer):
    """
    A TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
    
    TF-IDF improves upon simple count vectorization by:
    - Normalizing term frequencies within documents
    - Down-weighting terms that appear frequently across many documents
    - Emphasizing terms that are distinctive to specific documents
    """

    def __init__(self, tokenizer: Tokenizer, max_features: int = None, min_df: int = 1):
        """
        Initialize the TF-IDF vectorizer.
        
        Args:
            tokenizer: A Tokenizer instance to process text.
            max_features: Maximum number of features to keep (most important by IDF).
            min_df: Minimum document frequency - ignore terms appearing in fewer documents.
        """
        self._tokenizer = tokenizer
        self._max_features = max_features
        self._min_df = min_df
        self._vocabulary: Dict[str, int] = {}
        self._idf_values: Dict[str, float] = {}
        self._num_documents = 0

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        """
        Fit the vectorizer and transform texts to TF-IDF vectors.
        
        Args:
            texts: A list of text documents.
            
        Returns:
            A list of TF-IDF feature vectors (one per document).
        """
        # Tokenize all documents
        tokenized_docs = [self._tokenizer.tokenize(text) for text in texts]
        self._num_documents = len(tokenized_docs)
        
        # Build vocabulary and document frequencies
        doc_frequencies = Counter()
        all_tokens = set()
        
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            all_tokens.update(unique_tokens)
            doc_frequencies.update(unique_tokens)
        
        # Filter by min_df
        filtered_tokens = {
            token for token in all_tokens 
            if doc_frequencies[token] >= self._min_df
        }
        
        # Calculate IDF values
        self._idf_values = {
            token: math.log(self._num_documents / doc_frequencies[token])
            for token in filtered_tokens
        }
        
        # Select top features if max_features is set
        if self._max_features and len(self._idf_values) > self._max_features:
            # Sort by IDF value (descending) and take top features
            sorted_tokens = sorted(
                self._idf_values.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self._max_features]
            selected_tokens = {token for token, _ in sorted_tokens}
            self._idf_values = {
                token: idf for token, idf in self._idf_values.items()
                if token in selected_tokens
            }
        
        # Build vocabulary index
        self._vocabulary = {
            token: idx for idx, token in enumerate(sorted(self._idf_values.keys()))
        }
        
        # Transform documents
        return self._transform_docs(tokenized_docs)

    def fit(self, corpus: List[str]) -> None:
        """
        Fit the vectorizer on a corpus (learns vocabulary and IDF values).
        
        Args:
            corpus: A list of text documents.
        """
        # Just call fit_transform and discard the result
        self.fit_transform(corpus)

    def transform(self, texts: List[str]) -> List[List[float]]:
        """
        Transform texts to TF-IDF vectors using the fitted vocabulary.
        
        Args:
            texts: A list of text documents to transform.
            
        Returns:
            A list of TF-IDF feature vectors.
        """
        if not self._vocabulary:
            raise RuntimeError("Vectorizer has not been fitted. Call fit_transform first.")
        
        tokenized_docs = [self._tokenizer.tokenize(text) for text in texts]
        return self._transform_docs(tokenized_docs)

    def _transform_docs(self, tokenized_docs: List[List[str]]) -> List[List[float]]:
        """
        Internal method to transform tokenized documents to TF-IDF vectors.
        
        Args:
            tokenized_docs: List of tokenized documents.
            
        Returns:
            List of TF-IDF vectors.
        """
        vectors = []
        vocab_size = len(self._vocabulary)
        
        for tokens in tokenized_docs:
            # Initialize zero vector
            vector = [0.0] * vocab_size
            
            # Calculate term frequencies
            tf = Counter(tokens)
            doc_length = len(tokens)
            
            # Calculate TF-IDF for each term
            for token, count in tf.items():
                if token in self._vocabulary:
                    idx = self._vocabulary[token]
                    # TF: normalized term frequency
                    tf_value = count / doc_length if doc_length > 0 else 0
                    # IDF: inverse document frequency
                    idf_value = self._idf_values[token]
                    # TF-IDF
                    vector[idx] = tf_value * idf_value
            
            # L2 normalization
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            
            vectors.append(vector)
        
        return vectors

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names (tokens) in order.
        
        Returns:
            List of feature names.
        """
        if not self._vocabulary:
            return []
        
        # Sort by index to get correct order
        sorted_vocab = sorted(self._vocabulary.items(), key=lambda x: x[1])
        return [token for token, _ in sorted_vocab]
