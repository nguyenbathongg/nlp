"""
Count vectorization module for transforming text into numerical features.
"""

from typing import Dict, List

from src.core.interfaces import Tokenizer, Vectorizer


class CountVectorizer(Vectorizer):
    """
    Converts a collection of text documents to a matrix of token counts.
    
    This implementation creates a bag-of-words representation where each document
    is represented as a vector of token counts.
    
    Attributes:
        tokenizer (Tokenizer): The tokenizer used to split text into tokens
        vocabulary_ (Dict[str, int]): A mapping of tokens to their indices in the feature vector
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize the CountVectorizer with a tokenizer.
        
        Args:
            tokenizer (Tokenizer): A tokenizer that implements the Tokenizer interface, used to split text into tokens
        """
        self.tokenizer = tokenizer
        self.vocabulary_ = {} 
    
    def fit(self, corpus: List[str]) -> None:
        """
        Learns the vocabulary from a list of documents.
        
        Args:
            corpus (List[str]): A list of documents to learn the vocabulary from
        """
        # Initialize an empty set to hold unique tokens
        unique_tokens = set()
        
        # Process each document in the corpus
        for document in corpus:
            # Use the provided tokenizer to get tokens
            tokens = self.tokenizer.tokenize(document)
            
            # Add all tokens to the set to collect unique vocabulary
            unique_tokens.update(tokens)
        
        # Sort the unique tokens to ensure deterministic order
        sorted_tokens = sorted(unique_tokens)
        
        # Create vocabulary dictionary mapping tokens to indices
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted_tokens)}
    
    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms a list of documents into a list of count vectors.
        
        Args:
            documents (List[str]): A list of documents to transform
            
        Returns:
            List[List[int]]: A list of count vectors, where each vector
                            represents the word counts for a document
        
        Raises:
            ValueError: If the vocabulary hasn't been initialized by calling fit
        """
        if not self.vocabulary_:
            raise ValueError("Vocabulary not initialized. Call fit before transform.")
        
        result = []
        vocab_size = len(self.vocabulary_)
        
        for document in documents:
            # Create a zero vector with length equal to vocabulary size
            count_vector = [0] * vocab_size
            
            # Tokenize the document
            tokens = self.tokenizer.tokenize(document)
            
            # Count occurrences of each token
            for token in tokens:
                # Only count tokens that exist in our vocabulary
                if token in self.vocabulary_:
                    # Increment the count at the corresponding index
                    token_idx = self.vocabulary_[token]
                    count_vector[token_idx] += 1
            
            result.append(count_vector)
        
        return result