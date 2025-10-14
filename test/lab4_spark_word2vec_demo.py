from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.sql import functions as F

def main():
    # 1. Khởi tạo Spark Session
    # local[*] nghĩa là sử dụng tất cả các core CPU có sẵn trên máy local
    spark = (
    SparkSession.builder.appName("SparkWord2VecDemo")
    .master("local[*]")
    .config("spark.driver.memory", "4g")  # Tăng bộ nhớ driver nếu cần
    .getOrCreate()
)
    print("Spark Session đã được tạo.")

    # 2. Tải dữ liệu từ file nén gz
    # Spark có thể đọc trực tiếp từ file nén gz

    # Đọc file JSON từ file nén gz
    gz_path = "../data/c4-train.00000-of-01024-30K.json.gz"
    df = spark.read.json(gz_path)
    print("Dữ liệu đã được tải từ file nén gz vào DataFrame:")
    df.show(truncate=False)

    # 3. Tiền xử lý dữ liệu
    # Chọn cột 'text', chuyển thành chữ thường, và loại bỏ dấu câu
    processed_df = df.select(
        lower(regexp_replace('text', r'[^\w\s]', '')).alias('text')
    )
    
    # Sử dụng Tokenizer của Spark để tách câu thành các từ
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    tokenized_df = tokenizer.transform(processed_df)
    print("Dữ liệu sau khi tiền xử lý và token hóa:")
    tokenized_df.select("words").show(truncate=False)
    
    # 4. Cấu hình và huấn luyện model Word2Vec
    print("Bắt đầu huấn luyện Word2Vec model với Spark MLlib...")
    word2Vec = Word2Vec(vectorSize=100, minCount=1, inputCol="words", outputCol="result")
    model = word2Vec.fit(tokenized_df)
    print("Huấn luyện hoàn tất!")

    # 5. Sử dụng model
    # Tìm 5 từ đồng nghĩa nhất với "computer"
    synonyms = model.findSynonyms("computer", 5)
    print("\nTop 5 từ đồng nghĩa với 'computer':")
    synonyms.show()

    # 6. Dừng Spark session
    spark.stop()
    print("Spark Session đã đóng.")

if __name__ == "__main__":
    main()