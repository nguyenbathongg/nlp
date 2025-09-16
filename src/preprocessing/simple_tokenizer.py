import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    """
    A simple tokenizer that splits text on whitespace and handles basic punctuation.
    
    The tokenizer performs the following operations:
    1. Converts text to lowercase
    2. Splits text on whitespace
    3. Handles basic punctuation by separating them from words
    """
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into a list of tokens.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            List[str]: A list of tokens
        """
        if not text:
            return []
            
        # Convert to lowercase
        text = text.lower()
        
        # Insert spaces around punctuation to ensure they become separate tokens
        punctuation = ['.', ',', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}']
        for punct in punctuation:
            text = text.replace(punct, f" {punct} ")
        
        # Split on whitespace and filter out empty tokens
        tokens = [token for token in text.split() if token]
        
        return tokens