"""
Test file for Lab 2 - Count Vectorization
"""

import os
import sys
from typing import List

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing import RegexTokenizer
from src.representations import CountVectorizer


def print_matrix(matrix: List[List[int]], feature_names=None):
    """
    Simple print of document-term matrix vectors.
    
    Args:
        matrix: Document-term matrix where each inner list is a document vector
        feature_names: List of feature names (unused in this simplified version)
    """
    if not matrix:
        print("Empty matrix")
        return
    
    # Print the entire matrix as a list of vectors
    print(f"Complete matrix: {matrix}")
    
    # Also print each vector individually
    for i, doc_vector in enumerate(matrix):
        print(f"Document {i}: {doc_vector}")


def test_count_vectorizer():
    """
    Test the CountVectorizer implementation.
    """
    print("\n===== TESTING COUNT VECTORIZER =====")
    
    # Instantiate the RegexTokenizer from Lab 1
    tokenizer = RegexTokenizer(lower=True)
    
    # Instantiate the CountVectorizer with the tokenizer
    vectorizer = CountVectorizer(tokenizer)
    
    # Define a sample corpus
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    
    print("\n=== Sample Corpus ===")
    for i, doc in enumerate(corpus):
        print(f"Document {i}: {doc}")
    
    # Use fit and transform separately on the corpus
    vectorizer.fit(corpus)
    vectors = vectorizer.transform(corpus)
    
    # Get the learned vocabulary
    vocabulary = vectorizer.vocabulary_
    # Create our own feature_names list from the vocabulary
    feature_names = [""] * len(vocabulary)
    for token, idx in vocabulary.items():
        feature_names[idx] = token
    
    # Print the learned vocabulary
    print("\n=== Learned Vocabulary ===")
    print(f"Vocabulary size: {len(vocabulary)}")
    for token, idx in sorted(vocabulary.items(), key=lambda x: x[1]):
        print(f"Token: '{token}', Index: {idx}")
    
    # Print the resulting document-term matrix
    print("\n=== Document-Term Matrix ===")
    print_matrix(vectors, feature_names)


def main():
    """
    Main function to run the Lab 2 tests.
    """
    print("=" * 60)
    print("LAB 2: COUNT VECTORIZATION TEST")
    print("=" * 60)
    
    # Test count vectorizer
    test_count_vectorizer()
    


if __name__ == "__main__":
    main()