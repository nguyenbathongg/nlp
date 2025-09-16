# Week 2 Report: NLP Tokenization and Count Vectorization

## 1. Description of Work

### Lab 1: Tokenization

#### SimpleTokenizer Implementation

- Implemented a basic tokenizer that splits text on whitespace
- Added handling for basic punctuation by separating it from words
- Converted all text to lowercase for case-insensitive tokenization
- The tokenizer handles punctuation by inserting spaces around punctuation characters before splitting

#### RegexTokenizer Implementation

- Implemented a more sophisticated tokenizer using regular expressions
- Default pattern (`\w+|[^\w\s]`) captures both:
  - Word characters (letters, digits, underscore) in sequence
  - Individual non-word, non-whitespace characters (punctuation)
- Added option to enable/disable lowercase conversion
- More flexible than SimpleTokenizer as it uses regex pattern matching

### Lab 2: Count Vectorization

#### CountVectorizer Implementation

- Implemented a vectorizer that inherits from the Vectorizer interface
- Takes a tokenizer instance (from Lab 1) in the constructor
- Created a vocabulary mapping (dictionary of tokens to indices)
- Implemented the fit() method to learn vocabulary from a corpus
- Implemented the transform() method to convert documents into count vectors

#### Special Cases Handling

- Empty documents: Returns empty vectors with appropriate length
- Non-initialized vocabulary: Raises ValueError
- Unknown tokens: Ignored during transformation (only counts tokens that exist in the vocabulary)
- Preserves token order: Uses sorted vocabulary for consistent indices

## 2. Code Execution Results

### Lab 1: Tokenizers Testing

```
===== TESTING TOKENIZERS =====

----- Testing SimpleTokenizer -----

Sentence 1: Hello, world! This is a test.
Tokens (9): ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Sentence 2: NLP is fascinating... isn't it?
Tokens (11): ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Sentence 3: Let's see how it handles 123 numbers and punctuation!
Tokens (12): ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

----- Testing RegexTokenizer -----

Sentence 1: Hello, world! This is a test.
Tokens (9): ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Sentence 2: NLP is fascinating... isn't it?
Tokens (11): ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Sentence 3: Let's see how it handles 123 numbers and punctuation!
Tokens (12): ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
```

### Testing with UD_English-EWT Dataset

```
===== TESTING TOKENIZERS WITH DATASET =====

--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of ...
SimpleTokenizer Output (first 20 tokens): ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

### Lab 2: CountVectorizer Testing

```
===== TESTING COUNT VECTORIZER =====

=== Sample Corpus ===
Document 0: I love NLP.
Document 1: I love programming.
Document 2: NLP is a subfield of AI.

=== Learned Vocabulary ===
Vocabulary size: 10
Token: '.', Index: 0
Token: 'a', Index: 1
Token: 'ai', Index: 2
Token: 'i', Index: 3
Token: 'is', Index: 4
Token: 'love', Index: 5
Token: 'nlp', Index: 6
Token: 'of', Index: 7
Token: 'programming', Index: 8
Token: 'subfield', Index: 9

=== Document-Term Matrix ===
Complete matrix: [[1, 0, 0, 1, 0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]]
Document 0: [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
Document 1: [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
Document 2: [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

## 3. Analysis and Explanation of Results

### Tokenizer Comparison

1. **SimpleTokenizer vs RegexTokenizer**

   Despite using different approaches, both tokenizers produced similar results on the test sentences. However, they differ in their handling of hyphenated words:

   - **SimpleTokenizer** preserves hyphenated words as single tokens (e.g., 'al-zaman')
   - **RegexTokenizer** splits hyphenated words into separate tokens (e.g., 'al', '-', 'zaman')

   This difference is significant when processing real-world text, as seen in the UD_English-EWT dataset sample.

2. **Handling of Contractions**

   Both tokenizers handle contractions (e.g., "isn't", "Let's") in the same way, splitting them into separate tokens:

   - 'isn', "'", 't'
   - 'let', "'", 's'

   This approach is simple but potentially loses semantic meaning. More sophisticated tokenizers might preserve contractions or normalize them.

3. **Punctuation Handling**

   Both tokenizers treat each punctuation mark as a separate token, which is appropriate for basic NLP tasks but might need refinement for specific applications.

### CountVectorizer Analysis

1. **Vocabulary Construction**

   The vocabulary is constructed alphabetically, with each unique token assigned a unique index. This ensures deterministic behavior and allows for consistent vector creation.

2. **Document-Term Matrix**

   The resulting document-term matrix clearly shows the bag-of-words representation for each document:

   - Document 0 "I love NLP." contains the tokens: '.', 'i', 'love', 'nlp'
   - Document 1 "I love programming." contains: '.', 'i', 'love', 'programming'
   - Document 2 "NLP is a subfield of AI." contains: '.', 'a', 'ai', 'is', 'nlp', 'of', 'subfield'

   The binary patterns in the vectors make it easy to see shared terms between documents.

3. **Feature Representation**

   The count vectorizer successfully transforms text into numerical features that can be used for machine learning algorithms. Even with this small sample, we can observe:

   - Documents 0 and 1 share more features than either does with Document 2
   - The period ('.') appears in all documents
   - Some features like 'programming' and 'subfield' are unique to specific documents

## 4. Challenges and Solutions

### Tokenization Challenges

1. **Handling Special Cases**

   - Hyphenated words: As noted, the two tokenizers handle these differently
   - Multiple consecutive punctuation: Both tokenizers treat each punctuation mark separately
   - Solution: The RegexTokenizer provides flexibility through customizable patterns to address specific requirements

2. **Case Sensitivity**
   - Solution: Implemented lower=True parameter to control case conversion

### Count Vectorization Challenges

1. **Vocabulary Management**

   - Challenge: Creating a consistent mapping between tokens and indices
   - Solution: Used a sorted set of unique tokens to ensure deterministic ordering

2. **Efficient Vector Creation**

   - Challenge: Creating sparse vectors efficiently
   - Solution: Created zero vectors of appropriate length and updated only non-zero positions

3. **Interface Compatibility**
   - Challenge: Ensuring compatibility with the abstract Vectorizer interface
   - Solution: Properly implemented all required methods according to the interface specification
