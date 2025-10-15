# BÁO CÁO LAB 4: WORD EMBEDDINGS WITH WORD2VEC

**Sinh viên:** Nguyễn Bá Thông  
**Ngày thực hiện:** 14/10/2025

---

## 1. GIẢI THÍCH CÁC BƯỚC THỰC HIỆN

### 1.1. Task 1: Setup

**Các bước:**

1. **Cài đặt gensim:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Tải Pre-trained Model:**
   - Model: `glove-wiki-gigaword-50` (50-dimensional vectors)
   - Download tự động lần đầu qua `gensim.downloader.load()`
   - Kích thước: ~65MB

3. **Lý thuyết:**
   - Word embeddings: Dense vectors biểu diễn từ với semantic relationships
   - Word2Vec: "A word is characterized by the company it keeps"
   - Các từ tương tự nằm gần nhau trong vector space

### 1.2. Task 2: Word Embedding Exploration

**File:** `src/representations/word_embedder.py`

**Implement class `WordEmbedder` với 4 methods:**

1. **`__init__(self, model_name: str)`:** Load model bằng `gensim.downloader.load()`

2. **`get_vector(self, word: str)`:** Trả về embedding vector, handle OOV words

3. **`get_similarity(self, word1: str, word2: str)`:** Tính cosine similarity

4. **`get_most_similar(self, word: str, top_n: int = 10)`:** Tìm N từ tương tự nhất

### 1.3. Task 3: Document Embedding

**Method:** `embed_document(self, document: str, tokenizer) -> np.ndarray`

**Cách làm:**
1. Tokenize document
2. Lấy vectors cho mỗi token (bỏ qua OOV)
3. Tính mean của tất cả vectors
4. Nếu không có từ nào → zero vector

### 1.4. Evaluation

**File:** `test/lab4_test.py`

**4 operations theo yêu cầu:**
1. Get vector cho 'king'
2. Similarity: 'king'-'queen' và 'king'-'man'
3. 10 từ tương tự 'computer'
4. Embed câu "The queen rules the country."

---

## 2. HƯỚNG DẪN CHẠY CODE

### 2.1. Setup
```bash
cd d:\NLP\lab
pip install -r requirements.txt
```
**Lưu ý:** Model tự động download lần đầu (~65MB), cache tại `~/.gensim-data/`

### 2.2. Chạy Evaluation Test
```bash
python test/lab4_test.py
```

### 2.3. Bonus Task: Training Word2Vec
```bash
python test/lab4_embedding_training_demo.py
```
- Train trên UD_English-EWT corpus
- Save model vào `results/word2vec_ewt.model`

### 2.4. Advanced Task: Apache Spark
```bash
python test/lab4_spark_word2vec_demo.py
```
- Distributed training với PySpark MLlib
- Train trên C4 corpus

---

## 3. PHÂN TÍCH KẾT QUẢ

### 3.1. Nhận xét về độ tương đồng và từ đồng nghĩa

#### **Operation 1: Vector của 'king'**
- Shape: (50,) - 50 chiều như mong đợi
- Giá trị trong [-1, 1], normalized vectors
- Distributed representation: không interpret được từng dimension

#### **Operation 2: Similarity Scores**

| Cặp từ | Score | Giải thích |
|--------|-------|------------|
| king - queen | 0.7839 | Cao: cùng semantic field (hoàng gia), similar contexts |
| king - man | 0.5309 | Trung bình: có liên quan (is-a relationship) nhưng contexts khác |

**Kết luận:** Model capture được paradigmatic (king-queen) và syntagmatic (king-man) relationships

#### **Operation 3: 10 từ tương tự 'computer'**

| Rank | Word | Score | Relation |
|------|------|-------|----------|
| 1 | computers | 0.9165 | Morphological (plural) |
| 2 | software | 0.8815 | Part-of |
| 3 | technology | 0.8526 | Domain |
| 4 | electronic | 0.8126 | Attribute |
| 5 | internet | 0.8060 | Related tech |
| 6 | computing | 0.8026 | Morphological (verb) |
| 7 | devices | 0.8016 | Category |
| 8 | digital | 0.7992 | Attribute |
| 9 | applications | 0.7913 | Use-case |
| 10 | pc | 0.7883 | Synonym |

**Nhận xét:** Tất cả đều liên quan trực tiếp, đa dạng semantic relations (morphological, synonyms, attributes, domain)

### 3.2. Phân tích Document Embedding

**Input:** "The queen rules the country."

**Kết quả:**
- Tokens: ['the', 'queen', 'rules', 'the', 'country', '.']
- Valid tokens: 4/6 (66.7% coverage)
- OOV: ['.'] - punctuation được ignore

**Phân tích:**
- Document vector = mean của 4 word vectors
- Capture ý nghĩa: monarchy, government, leadership
- Shape: (50,) giống word vectors

**Limitations:**
- Không có word weighting (function words = content words)
- Không có word order (bag-of-words)
- Single vector cho complex document

**Kết luận:** Simple averaging hiệu quả cho short texts, cần methods phức tạp hơn cho long documents

### 3.3. So sánh Pre-trained vs Self-trained Model

| Tiêu chí | Pre-trained (GloVe) | Self-trained (Word2Vec) |
|----------|---------------------|-------------------------|
| **Corpus** | Wikipedia + Gigaword (6B tokens) | UD_English-EWT (254K tokens) |
| **Vocabulary** | 400K words | ~5K words |
| **Training Time** | Days | Minutes |
| **OOV Rate** | ~5% | ~30-40% |
| **Use Cases** | General NLP | Domain-specific |

**Khi nào dùng Pre-trained:** General tasks, large vocabulary, no training needed  
**Khi nào dùng Self-trained:** Domain-specific jargon, specialized terminology

---

## 4. KHÓ KHĂN VÀ GIẢI PHÁP

### 4.1. Vấn đề tương thích thư viện

**Khó khăn:** Python 3.13.6 quá mới, gensim 4.x chưa có prebuilt wheel, gensim 0.10.1 incompatible với scipy mới

**Giải pháp:**
- Dùng Python 3.11 (có prebuilt wheels)
- Hoặc install Visual C++ Build Tools để compile from source
- Hoặc dùng Conda

### 4.2. Memory Management

**Khó khăn:** 300d model (~1GB) crash trên RAM < 4GB, git repository bloat

**Giải pháp:**
- Dùng 50d model (~65MB) thay vì 300d
- Thêm `.embeddings_cache/` vào `.gitignore`
- Model tự động cache tại `~/.gensim-data/`

### 4.3. Out-of-Vocabulary (OOV) Words

**Khó khăn:** Domain terms, proper nouns, punctuation không có trong vocabulary → ảnh hưởng document embedding

**Giải pháp:**
- Return None cho OOV words, skip khi averaging
- Preprocessing: lowercase, remove punctuation
- Fallback: thử lowercase, remove punctuation trước khi give up
- Future: dùng FastText (subword embeddings)


## 5. TRÍCH DẪN TÀI LIỆU

1. **Word2Vec:** Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"
2. **GloVe:** Pennington et al. (2014) - https://nlp.stanford.edu/projects/glove/
