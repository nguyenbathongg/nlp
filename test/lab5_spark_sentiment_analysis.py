"""
Lab 5: Sentiment Analysis with PySpark ML Pipeline
Demonstrates text classification using Apache Spark MLlib
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

def main():
    # 1. Initialize Spark Session
    print("Step 1: Initializing Spark Session...")
    
    # Set environment variables for Windows compatibility
    os.environ['PYSPARK_PYTHON'] = 'python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
    
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark Session initialized")
    print()
    
    # 2. Load Data
    print("Step 2: Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiments.csv')
    data_path = os.path.abspath(data_path)
    
    # Read CSV without inferSchema to avoid casting issues
    df = spark.read.csv(data_path, header=True, inferSchema=False)
    
    # Show initial data info
    initial_row_count = df.count()
    print(f"Initial dataset: {initial_row_count} rows")
    
    # Filter valid sentiment values (must be numeric: -1, 0, or 1)
    from pyspark.sql.functions import col, regexp_extract
    df = df.filter(col("sentiment").rlike("^-?[0-9]+$"))
    df = df.filter(col("text").isNotNull())
    
    cleaned_row_count = df.count()
    print(f"After cleaning: {cleaned_row_count} rows")
    
    # Convert sentiment to integer, then normalize to 0/1/2
    # -1 (Bearish) -> 0, 0 (Neutral) -> 1, 1 (Bullish) -> 2
    df = df.withColumn("sentiment_int", col("sentiment").cast("integer"))
    df = df.withColumn("label", col("sentiment_int") + 1)
    df = df.select("text", "label")
    
    print("Data loaded and labels normalized")
    df.groupBy("label").count().orderBy("label").show()
    
    # 3. Split data into training and test sets (80/20)
    print("Step 3: Splitting data...")
    trainingData, testData = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set: {trainingData.count()} samples")
    print(f"Test set: {testData.count()} samples")
    print()
    
    # 4. Build Preprocessing Pipeline
    print("Step 4: Building ML Pipeline...")
    
    # Tokenizer: Splits text into words
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    print("Tokenizer configured")
    
    # StopWordsRemover: Removes common stop words
    stopwordsRemover = StopWordsRemover(
        inputCol="words", 
        outputCol="filtered_words"
    )
    print("StopWordsRemover configured")
    
    # HashingTF: Converts tokens to fixed-size feature vector
    hashingTF = HashingTF(
        inputCol="filtered_words", 
        outputCol="raw_features", 
        numFeatures=10000
    )
    print("HashingTF configured (10,000 features)")
    
    # IDF: Inverse Document Frequency
    idf = IDF(
        inputCol="raw_features", 
        outputCol="features"
    )
    print("IDF configured")
    
    # 5. Configure Logistic Regression Model
    print()
    print("Step 5: Configuring Logistic Regression...")
    lr = LogisticRegression(
        maxIter=10, 
        regParam=0.001, 
        featuresCol="features", 
        labelCol="label"
    )
    print("LogisticRegression configured (maxIter=10, regParam=0.001)")
    print()
    
    # 6. Assemble Pipeline
    print("Step 6: Assembling pipeline...")
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
    print("Pipeline assembled with 5 stages")
    print()
    
    # 7. Train the Model
    print("Step 7: Training model...")
    print("This may take a few minutes...")
    model = pipeline.fit(trainingData)
    print("Training complete!")
    print()
    
    # 8. Make Predictions
    print("Step 8: Making predictions on test set...")
    predictions = model.transform(testData)
    print("Predictions complete")
    print()
    
    # Show sample predictions
    print("Sample predictions (first 10):")
    print("-" * 70)
    predictions.select("text", "label", "prediction") \
        .limit(10) \
        .show(truncate=50)
    
    # 9. Evaluate the Model
    print("Step 9: Evaluating model...")
    print("-" * 70)
    
    # Accuracy
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator_accuracy.evaluate(predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # F1 Score
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)
    print(f"F1 Score: {f1:.4f}")
    
    # Weighted Precision
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    precision = evaluator_precision.evaluate(predictions)
    print(f"Weighted Precision: {precision:.4f}")
    
    # Weighted Recall
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    recall = evaluator_recall.evaluate(predictions)
    print(f"Weighted Recall: {recall:.4f}")
    print()
    
    # Confusion Matrix (manual calculation)
    print("Confusion Matrix:")
    print("-" * 70)
    predictions.groupBy("label", "prediction").count() \
        .orderBy("label", "prediction") \
        .show()
    
    print("=" * 70)
    print("Lab 5 Spark ML Pipeline Complete!")
    print("=" * 70)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()
