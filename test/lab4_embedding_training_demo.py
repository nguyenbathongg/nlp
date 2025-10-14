import sys
import os
from gensim.models import Word2Vec
from src.preprocessing.regex_tokenizer import RegexTokenizer

class SentenceStream:
    """
    A memory-friendly iterator to stream sentences from a large text file.
    It reads the file line by line and tokenizes each line.
    """
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    yield self.tokenizer.tokenize(line)

def main():
    # 1. Define paths
    dataset_path = "./UD_English-EWT/en_ewt-ud-train.txt"
    output_model_path = "./results/word2vec_ewt.model"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    # 2. Create a sentence streamer for memory-efficient processing
    print("\nSetting up sentence streamer...")
    tokenizer = RegexTokenizer()
    sentences = SentenceStream(dataset_path, tokenizer)

    # 3. Train the Word2Vec model
    print("\nTraining Word2Vec model... (This may take a few minutes)")
    # Parameters:
    # vector_size: Dimensionality of the word vectors.
    # window: Maximum distance between the current and predicted word within a sentence.
    # min_count: Ignores all words with a total frequency lower than this.
    # workers: Use these many worker threads to train the model (=faster training).
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4, epochs=1) # Reduced epochs for quick demo
    print("Training complete.")

    # 4. Save the model
    print(f"\nSaving model to: {output_model_path}")
    model.save(output_model_path)
    print("Model saved.")

    # 5. Demonstrate the trained model
    print("\n--- Demonstrating Model Capabilities ---")
    
    # Find most similar words
    try:
        similar_words = model.wv.most_similar('king', topn=5)
        print("\nWords most similar to 'king':")
        for word, score in similar_words:
            print(f"  - {word}: {score:.4f}")
    except KeyError:
        print("\n'king' not in vocabulary (or vocabulary is too small). Try another word like 'president' or 'man'.")

    # Perform analogy task
    try:
        analogy = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
        print("\nAnalogy 'king - man + woman':")
        print(f"  -> {analogy[0][0]} (similarity: {analogy[0][1]:.4f})")
    except KeyError as e:
        print(f"\nCould not perform analogy, a word was not in the vocabulary: {e}")


if __name__ == "__main__":
    main()