"""
Test file for Lab 1 - Tokenization
"""

import os
import sys
from typing import List

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing import SimpleTokenizer, RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data


def test_tokenizers():
    """
    Test the tokenizer implementations with example sentences.
    """
    print("\n===== TESTING TOKENIZERS =====")
    
    # Test sentences
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]
    
    # Create tokenizer instances
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer(lower=True)
    
    # Test SimpleTokenizer
    print("\n----- Testing SimpleTokenizer -----")
    for i, sentence in enumerate(sentences, 1):
        tokens = simple_tokenizer.tokenize(sentence)
        print(f"\nSentence {i}: {sentence}")
        print(f"Tokens ({len(tokens)}): {tokens}")
    
    # Test RegexTokenizer
    print("\n----- Testing RegexTokenizer -----")
    for i, sentence in enumerate(sentences, 1):
        tokens = regex_tokenizer.tokenize(sentence)
        print(f"\nSentence {i}: {sentence}")
        print(f"Tokens ({len(tokens)}): {tokens}")
    
    return simple_tokenizer, regex_tokenizer


def test_with_dataset():
    """
    Test tokenizers with sample text from a dataset.
    """
    print("\n===== TESTING TOKENIZERS WITH DATASET =====")
    
    # Create tokenizer instances
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer(lower=True)
    
    # Load the raw text from the UD_English-EWT dataset
    dataset_path = "UD_English-EWT/en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    
    # Take a small portion of the text for demonstration
    sample_text = raw_text[:500]  # First 500 characters
    
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")
    
    # Tokenize with SimpleTokenizer
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    
    # Tokenize with RegexTokenizer
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")


def main():
    """
    Main function to run the Lab 1 tests.
    """
    print("=" * 60)
    print("LAB 1: TOKENIZATION TEST")
    print("=" * 60)
    
    # Test tokenizers with example sentences
    test_tokenizers()
    
    # Test tokenizers with dataset
    test_with_dataset()


if __name__ == "__main__":
    main()