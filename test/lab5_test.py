"""
Lab 5 Test: Text Classification with Logistic Regression
Tests the TextClassifier on sentiment analysis task with Twitter Financial News Dataset
"""

from sklearn.model_selection import train_test_split
from datasets import load_dataset
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.models.text_classifier import TextClassifier


def main():
    # Load Twitter Financial News Sentiment Dataset
    print("Loading dataset from Hugging Face...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    print("Dataset loaded!")
    print()
    
    # Extract texts and labels from training split
    train_data = ds['train']
    texts = train_data['text']
    labels = train_data['label']
    
    print(f"Dataset: {len(texts)} tweets")
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution: {label_counts}")
    print()

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()

    # Initialize components
    print("Initializing components...")
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)
    classifier = TextClassifier(vectorizer)
    print("RegexTokenizer initialized")
    print("CountVectorizer initialized")
    print("TextClassifier initialized")
    print()

    # Train the classifier
    print("Training classifier...")
    classifier.fit(X_train, y_train)
    print("Training complete!")
    print()

    # Make predictions on test set
    print("Making predictions on test set...")
    y_pred = classifier.predict(X_test)
    print()

    # Display test results
    print("TEST RESULTS (first 10 samples):")
    print("-" * 60)
    for i, (text, true_label, pred_label) in enumerate(zip(X_test[:10], y_test[:10], y_pred[:10])):
        sentiment_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
        sentiment_true = sentiment_map.get(true_label, f"Unknown({true_label})")
        sentiment_pred = sentiment_map.get(pred_label, f"Unknown({pred_label})")
        match = "CORRECT" if true_label == pred_label else "WRONG"
        print(f"[{match}] Text: \"{text[:60]}...\"")
        print(f"  True: {sentiment_true}, Predicted: {sentiment_pred}")
        print()

    # Evaluate predictions
    print("EVALUATION METRICS:")
    print("-" * 60)
    
    # For multi-class, we need to use different average settings
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall:    {recall:.4f} (weighted)")
    print(f"F1 Score:  {f1:.4f} (weighted)")
    print()

    # Test on new, unseen texts
    print("TESTING ON NEW FINANCIAL TWEETS:")
    print("-" * 60)
    new_texts = [
        "Stock prices are soaring! Great news for investors.",
        "Market crash imminent, sell everything now!",
        "Company reports quarterly earnings, no significant change."
    ]
    
    new_predictions = classifier.predict(new_texts)
    sentiment_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    
    for text, pred in zip(new_texts, new_predictions):
        sentiment = sentiment_map.get(pred, f"Unknown({pred})")
        print(f"Text: \"{text}\"")
        print(f"Prediction: {sentiment}")
        print()



if __name__ == "__main__":
    main()
