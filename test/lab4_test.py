import sys
import os
import pprint
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.word_embedder import WordEmbedder

def main():
    """
    Main function to test the WordEmbedder.
    """
    print("--- WordEmbedder Evaluation ---")
    print("NOTE: The first time this runs, it will download the GloVe model (~65MB).")
    print("This may take a few minutes...")

    try:
        # 1. Instantiate the embedder
        embedder = WordEmbedder(model_name='glove-wiki-gigaword-50')
        tokenizer = RegexTokenizer()

        # 2. Get vector for a word
        print("\n--- Get Vector ---")
        king_vector = embedder.get_vector('king')
        print(f"Vector for 'king' (first 5 dims): {king_vector[:5]}")
        print(f"Vector dimension: {len(king_vector)}")

        # 3. Get word similarity
        print("\n--- Get Similarity ---")
        sim_kq = embedder.get_similarity('king', 'queen')
        sim_km = embedder.get_similarity('king', 'man')
        print(f"Similarity('king', 'queen'): {sim_kq:.4f}")
        print(f"Similarity('king', 'man'):   {sim_km:.4f}")

        # 4. Get most similar words
        print("\n--- Get Most Similar ---")
        most_similar_computer = embedder.get_most_similar('computer')
        print("Most similar to 'computer':")
        pprint.pprint(most_similar_computer)

        # 5. Embed a document
        print("\n--- Embed Document ---")
        doc = "The queen rules the country."
        doc_vector = embedder.embed_document(doc, tokenizer)
        print(f"Original document: '{doc}'")
        print(f"Document vector (first 5 dims): {doc_vector[:5]}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This might be due to a network issue while downloading the model.")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()