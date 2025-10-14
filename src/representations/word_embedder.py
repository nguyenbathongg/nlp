import numpy as np
import gensim.downloader as api
from typing import List
from src.core.interfaces import Tokenizer

class WordEmbedder:
    """
    A class to handle loading and using pre-trained word embeddings.
    """

    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        """
        Loads a pre-trained model from gensim's repository.

        Args:
            model_name: The name of the model to load.
        """
        try:
            self.model = api.load(model_name)
            self.vector_size = self.model.vector_size
        except ValueError as e:
            print(f"Error loading model: {e}")
            print(f"Please choose from available models: {list(api.info()['models'].keys())}")
            raise

    def get_vector(self, word: str) -> np.ndarray:
        """
        Gets the vector for a single word.

        Args:
            word: The word to get the vector for.

        Returns:
            A numpy array representing the word's embedding.
            Returns a zero vector if the word is out of vocabulary (OOV).
        """
        try:
            return self.model[word]
        except KeyError:
            # Handle Out-of-Vocabulary (OOV) words
            return np.zeros(self.vector_size)

    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Calculates the cosine similarity between two words.

        Returns:
            The cosine similarity score (float).
        """
        return self.model.similarity(word1, word2)

    def get_most_similar(self, word: str, top_n: int = 10) -> List[tuple[str, float]]:
        """
        Finds the most similar words to a given word.

        Returns:
            A list of (word, similarity_score) tuples.
        """
        return self.model.most_similar(word, topn=top_n)

    def embed_document(self, document: str, tokenizer: Tokenizer) -> np.ndarray:
        """
        Creates a document embedding by averaging the vectors of its words.

        Args:
            document: The document string to embed.
            tokenizer: A tokenizer instance to split the document.

        Returns:
            A numpy array representing the document's embedding.
        """
        tokens = tokenizer.tokenize(document)
        word_vectors = [self.get_vector(token) for token in tokens]
        
        # Filter out zero vectors for OOV words before averaging
        valid_vectors = [vec for vec in word_vectors if np.any(vec)]

        if not valid_vectors:
            return np.zeros(self.vector_size)
        
        # Return the element-wise mean of the vectors
        return np.mean(valid_vectors, axis=0)