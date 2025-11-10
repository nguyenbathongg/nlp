"""
Lab 5 Model Improvement: Testing different approaches to improve text classification
Experiments:
1. Baseline: CountVectorizer + Logistic Regression
2. TF-IDF: TfidfVectorizer + Logistic Regression  
3. Word2Vec: Word embeddings + Logistic Regression
4. Alternative Model: TfidfVectorizer + Naive Bayes
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.representations.tfidf_vectorizer import TfidfVectorizer
from src.representations.word_embedder import WordEmbedder
from src.models.text_classifier import TextClassifier


def print_metrics(name, accuracy, precision, recall, f1):
    """Print evaluation metrics in formatted way"""
    print(f"{name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print()


def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


class Word2VecClassifier:
    """
    Classifier using Word2Vec embeddings for feature extraction
    """
    def __init__(self, tokenizer, model_name='glove-wiki-gigaword-50'):
        self.tokenizer = tokenizer
        self.embedder = WordEmbedder(model_name)
        self.model = None
        
    def fit(self, texts, labels):
        """Train classifier on Word2Vec features"""
        from sklearn.linear_model import LogisticRegression
        # Tokenize and get document embeddings (average of word vectors)
        X = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            if tokens:
                # Get vectors for each token
                vectors = [self.embedder.get_vector(token) for token in tokens 
                          if self.embedder.get_vector(token) is not None]
                if vectors:
                    # Average the vectors
                    doc_vector = np.mean(vectors, axis=0)
                else:
                    doc_vector = np.zeros(self.embedder.vector_size)
            else:
                doc_vector = np.zeros(self.embedder.vector_size)
            X.append(doc_vector)
        
        self.model = LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000)
        self.model.fit(X, labels)
        
    def predict(self, texts):
        """Predict using Word2Vec features"""
        X = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            if tokens:
                vectors = [self.embedder.get_vector(token) for token in tokens
                          if self.embedder.get_vector(token) is not None]
                if vectors:
                    doc_vector = np.mean(vectors, axis=0)
                else:
                    doc_vector = np.zeros(self.embedder.vector_size)
            else:
                doc_vector = np.zeros(self.embedder.vector_size)
            X.append(doc_vector)
        
        return self.model.predict(X).tolist()


class NaiveBayesClassifier:
    """
    Naive Bayes classifier wrapper (needs non-negative features)
    """
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.model = None
        
    def fit(self, texts, labels):
        """Train Naive Bayes classifier"""
        X = self.vectorizer.fit_transform(texts)
        # Convert to non-negative (Naive Bayes requirement)
        X_array = np.array(X)
        # Shift to make all values non-negative
        X_min = X_array.min()
        if X_min < 0:
            X_array = X_array - X_min
        self.model = MultinomialNB()
        self.model.fit(X_array, labels)
        
    def predict(self, texts):
        """Predict with Naive Bayes"""
        X = self.vectorizer.transform(texts)
        X_array = np.array(X)
        # Apply same shift as training
        X_min = X_array.min()
        if X_min < 0:
            X_array = X_array - X_min
        return self.model.predict(X_array).tolist()


def main():
    print("LAB 5: MODEL IMPROVEMENT EXPERIMENTS")
    
    # Load dataset
    print("Loading Twitter Financial News dataset...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    train_data = ds['train']
    
    texts = list(train_data['text'])
    labels = list(train_data['label'])
    
    print(f"Dataset loaded: {len(texts)} samples")
    print(f"  Label distribution: Bearish={labels.count(0)}, Bullish={labels.count(1)}, Neutral={labels.count(2)}")
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    
    # Store results for comparison
    results = {}
    
    # ========================================================================
    # EXPERIMENT 1: Baseline (CountVectorizer + Logistic Regression)
    # ========================================================================
    print("EXPERIMENT 1: Baseline (CountVectorizer + LR)")
    print("Training baseline model...")
    
    tokenizer1 = RegexTokenizer()
    vectorizer1 = CountVectorizer(tokenizer1)
    classifier1 = TextClassifier(vectorizer1)
    
    classifier1.fit(X_train, y_train)
    y_pred1 = classifier1.predict(X_test)
    metrics1 = evaluate_model(y_test, y_pred1)
    results['Baseline (Count + LR)'] = metrics1
    
    print_metrics("Baseline Results", metrics1['accuracy'], metrics1['precision'], 
                  metrics1['recall'], metrics1['f1'])
    
    # ========================================================================
    # EXPERIMENT 2: TF-IDF Vectorizer + Logistic Regression
    # ========================================================================
    print("EXPERIMENT 2: TF-IDF Improvement")
    print("Training with TF-IDF vectorizer...")
    print("Expected improvement: TF-IDF down-weights common terms and")
    print("emphasizes distinctive terms, improving classification.")
    print()
    
    tokenizer2 = RegexTokenizer()
    vectorizer2 = TfidfVectorizer(tokenizer2, max_features=5000, min_df=2)
    classifier2 = TextClassifier(vectorizer2)
    
    classifier2.fit(X_train, y_train)
    y_pred2 = classifier2.predict(X_test)
    metrics2 = evaluate_model(y_test, y_pred2)
    results['TF-IDF + LR'] = metrics2
    
    print_metrics("TF-IDF Results", metrics2['accuracy'], metrics2['precision'],
                  metrics2['recall'], metrics2['f1'])
    
    improvement = (metrics2['f1'] - metrics1['f1']) / metrics1['f1'] * 100
    print(f"Improvement over baseline: {improvement:+.2f}%")
    
    # ========================================================================
    # EXPERIMENT 3: Word2Vec Embeddings + Logistic Regression
    # ========================================================================
    print("EXPERIMENT 3: Word2Vec Embeddings")
    print("Training with Word2Vec embeddings from Lab 4...")
    print("Expected improvement: Dense embeddings capture semantic meaning")
    print("better than sparse bag-of-words representations.")
    print()
    print("Loading GloVe model (this may take a minute)...")
    
    tokenizer3 = RegexTokenizer()
    classifier3 = Word2VecClassifier(tokenizer3, 'glove-wiki-gigaword-50')
    
    print("Training Word2Vec classifier...")
    classifier3.fit(X_train, y_train)
    y_pred3 = classifier3.predict(X_test)
    metrics3 = evaluate_model(y_test, y_pred3)
    results['Word2Vec + LR'] = metrics3
    
    print_metrics("Word2Vec Results", metrics3['accuracy'], metrics3['precision'],
                  metrics3['recall'], metrics3['f1'])
    
    improvement = (metrics3['f1'] - metrics1['f1']) / metrics1['f1'] * 100
    print(f"Improvement over baseline: {improvement:+.2f}%")
    
    # ========================================================================
    # EXPERIMENT 4: TF-IDF + Naive Bayes
    # ========================================================================
    print("EXPERIMENT 4: Naive Bayes Classifier")
    print("Training Naive Bayes with TF-IDF...")
    print("Naive Bayes is a probabilistic classifier that often works well")
    print("for text classification tasks.")
    print()
    
    tokenizer4 = RegexTokenizer()
    vectorizer4 = TfidfVectorizer(tokenizer4, max_features=5000, min_df=2)
    classifier4 = NaiveBayesClassifier(vectorizer4)
    
    classifier4.fit(X_train, y_train)
    y_pred4 = classifier4.predict(X_test)
    metrics4 = evaluate_model(y_test, y_pred4)
    results['TF-IDF + Naive Bayes'] = metrics4
    
    print_metrics("Naive Bayes Results", metrics4['accuracy'], metrics4['precision'],
                  metrics4['recall'], metrics4['f1'])
    
    improvement = (metrics4['f1'] - metrics1['f1']) / metrics1['f1'] * 100
    print(f"Improvement over baseline: {improvement:+.2f}%")
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("FINAL COMPARISON - ALL EXPERIMENTS")
    
    print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 78)
    
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print()
    print(f"Best Model: {best_model[0]}")
    print(f"   F1 Score: {best_model[1]['f1']:.4f}")
    
    # Analysis
    print()
    print("ANALYSIS:")
    print("-" * 70)
    print("1. TF-IDF vs Count Vectorizer:")
    print("   TF-IDF typically improves performance by down-weighting common")
    print("   terms and emphasizing distinctive features.")
    print()
    print("2. Word2Vec Embeddings:")
    print("   Dense embeddings can capture semantic relationships but may")
    print("   require more data to outperform simpler methods.")
    print()
    print("3. Naive Bayes:")
    print("   Often fast and effective for text, especially with TF-IDF.")
    print("   May work better on balanced datasets.")
    print()



if __name__ == "__main__":
    main()
