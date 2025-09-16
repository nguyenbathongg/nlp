from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    """
    Abstract base class for text tokenizers.
    Tokenizers convert text strings into lists of tokens.
    """
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into a list of tokens.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            List[str]: A list of tokens
        """
        pass


class Vectorizer(ABC):
    """
    Abstract base class for text vectorizers.
    Vectorizers convert documents (text strings) into numerical vectors
    using the bag-of-words model.
    """
    
    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """
        Learns the vocabulary from a list of documents (corpus).
        
        Args:
            corpus (List[str]): A list of documents to learn the vocabulary from
        """
        pass
    
    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms a list of documents into a list of count vectors
        based on the learned vocabulary.
        
        Args:
            documents (List[str]): A list of documents to transform
            
        Returns:
            List[List[int]]: A list of count vectors, where each vector
                represents the word counts for a document
        """
        pass
    
