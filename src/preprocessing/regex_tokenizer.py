import re
from typing import List
from src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    '''
    A regex-based tokenizer that uses regular expressions to extract tokens from text.
    
    This tokenizer uses a single regular expression to identify tokens, which can be more robust
    than simple splitting. It captures words (sequences of word characters) and individual 
    non-word, non-whitespace characters.
    
    Default pattern captures:
    - Word characters (letters, digits, underscore) in sequence
    - Single characters that are not word characters or whitespace
    '''
    
    def __init__(self, pattern=r"\w+|[^\w\s]", lower=True):
        '''
        Initialize the RegexTokenizer with a regex pattern.
        
        Args:
            pattern (str): Regular expression pattern to use for tokenization
                Default pattern matches word characters in sequence and 
                individual non-word, non-whitespace characters
            lower (bool): Whether to convert tokens to lowercase (default: True)
        '''
        self.pattern = re.compile(pattern)
        self.lower = lower
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into a list of tokens using regex.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            List[str]: A list of tokens
        """
        if not text:
            return []
        
        # Find all matches of the pattern in the text
        tokens = self.pattern.findall(text)
        
        # Convert to lowercase if requested
        if self.lower:
            tokens = [token.lower() for token in tokens]
            
        return tokens