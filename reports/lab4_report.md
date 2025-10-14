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

**Mục tiêu:** Tạo class `WordEmbedder` để khám phá word embeddings.

**Các bước thực hiện:**

1. **Tạo file `src/representations/word_embedder.py`**

2. **Implement class `WordEmbedder`:**

   a) **Constructor `__init__(self, model_name: str)`:**
      ```python
      def __init__(self, model_name: str):
          """
          Khởi tạo WordEmbedder với pre-trained model
          
          Args:
              model_name: Tên model từ gensim downloader
                         (ví dụ: 'glove-wiki-gigaword-50')
          """
          self.model = gensim.downloader.load(model_name)
          self.model_name = model_name
      ```

   b) **Method `get_vector(self, word: str)`:**
      - Trả về embedding vector của một từ
      - Xử lý Out-of-Vocabulary (OOV) words: trả về None hoặc zero vector
      - Handle case-sensitivity
      ```python
      def get_vector(self, word: str) -> np.ndarray:
          if word in self.model:
              return self.model[word]
          return None  # Hoặc np.zeros(dimension)
      ```

   c) **Method `get_similarity(self, word1: str, word2: str)`:**
      - Tính cosine similarity giữa vectors của hai từ
      - Công thức: similarity = (v1 · v2) / (||v1|| × ||v2||)
      ```python
      def get_similarity(self, word1: str, word2: str) -> float:
          return self.model.similarity(word1, word2)
      ```

   d) **Method `get_most_similar(self, word: str, top_n: int = 10)`:**
      - Sử dụng built-in method `most_similar()` của model
      - Trả về list các (word, similarity_score) tuples
      - Top N từ tương tự nhất
      ```python
      def get_most_similar(self, word: str, top_n: int = 10):
          return self.model.most_similar(word, topn=top_n)
      ```

### 1.3. Task 3: Document Embedding

**Mục tiêu:** Tạo vector representation cho cả document bằng cách average word vectors.

**Phương pháp:**
- Document embedding = Mean của tất cả word vectors trong document
- Đây là baseline approach đơn giản nhưng hiệu quả

**Các bước thực hiện:**

1. **Implement method `embed_document(self, document: str, tokenizer)`:**

   a) **Input:**
      - `document`: String cần embed
      - `tokenizer`: Tokenizer object từ Lab 1 (SimpleTokenizer hoặc RegexTokenizer)

   b) **Xử lý:**
      ```python
      def embed_document(self, document: str, tokenizer) -> np.ndarray:
          # Bước 1: Tokenize document
          tokens = tokenizer.tokenize(document)
          
          # Bước 2: Lấy vectors cho mỗi token
          vectors = []
          for token in tokens:
              vec = self.get_vector(token)
              if vec is not None:  # Bỏ qua OOV words
                  vectors.append(vec)
          
          # Bước 3: Xử lý edge case
          if len(vectors) == 0:
              # Không có từ nào trong vocabulary
              return np.zeros(self.model.vector_size)
          
          # Bước 4: Tính mean vector
          return np.mean(vectors, axis=0)
      ```

   c) **Đặc điểm:**
      - OOV words được ignore (không ảnh hưởng đến kết quả)
      - Nếu document toàn OOV words → zero vector
      - Vector kết quả có cùng dimension với word vectors (50 chiều)

### 1.4. Evaluation

**Mục tiêu:** Tạo test file để demonstrate các operations.

**Các bước:**

1. **Tạo file `test/lab4_test.py`**

2. **Implement các operations theo yêu cầu:**

   ```python
   # 1. Get vector for 'king'
   king_vec = embedder.get_vector('king')
   print(f"Vector for 'king': {king_vec}")
   print(f"Shape: {king_vec.shape}")
   
   # 2. Get similarity
   sim_kq = embedder.get_similarity('king', 'queen')
   sim_km = embedder.get_similarity('king', 'man')
   print(f"Similarity (king, queen): {sim_kq}")
   print(f"Similarity (king, man): {sim_km}")
   
   # 3. Get 10 most similar to 'computer'
   similar = embedder.get_most_similar('computer', top_n=10)
   print(f"10 most similar to 'computer':")
   for word, score in similar:
       print(f"  {word}: {score:.4f}")
   
   # 4. Embed sentence
   doc_vec = embedder.embed_document(
       "The queen rules the country.",
       tokenizer
   )
   print(f"Document vector shape: {doc_vec.shape}")
   print(f"Document vector: {doc_vec}")
   ```

3. **Output mong đợi:**
   - Vector 'king': array 50 phần tử
   - Similarity scores: float values (0-1)
   - 10 từ tương tự: danh sách ranked theo similarity
   - Document vector: array 50 phần tử (mean của word vectors)

---

## 2. HƯỚNG DẪN CHẠY CODE

### 2.1. Setup môi trường

**Bước 1: Cài đặt dependencies**
```bash
# Di chuyển vào thư mục project
cd d:\NLP\lab

# Cài đặt tất cả dependencies (bao gồm gensim)
pip install -r requirements.txt
```

**Lưu ý:**
- Lần đầu tiên chạy WordEmbedder, model `glove-wiki-gigaword-50` sẽ được tự động download
- File download khoảng 65MB, có thể mất vài phút tuỳ tốc độ internet
- Model được cache tại `~/.gensim-data/` để không phải download lại

### 2.2. Chạy Evaluation Test

```bash
# Chạy test file chính
python test/lab4_test.py
```

**Output mong đợi:**

```
Loading model 'glove-wiki-gigaword-50'...
Model loaded successfully!

=== Operation 1: Get vector for 'king' ===
Vector shape: (50,)
First 10 elements: [0.50451 0.68607 -0.59517 ...]

=== Operation 2: Similarity scores ===
Similarity (king, queen): 0.7839
Similarity (king, man): 0.5309

=== Operation 3: 10 most similar to 'computer' ===
1. computers        0.9165
2. software         0.8815
3. technology       0.8526
...

=== Operation 4: Embed document ===
Document: "The queen rules the country."
Tokens: ['the', 'queen', 'rules', 'the', 'country', '.']
Document vector shape: (50,)
```

### 2.3. Chạy Bonus Task: Training Word2Vec from Scratch

**Yêu cầu:**
- Dataset: `UD_English-EWT/en_ewt-ud-train.txt`
- Gensim 4.x 

**Chạy script:**
```bash
python test/lab4_embedding_training_demo.py
```

**Chức năng của script:**
1. **Stream Data:** Đọc raw text từ UD_English-EWT dataset một cách memory-efficient
2. **Train Model:** Sử dụng gensim Word2Vec để train model mới
3. **Save Model:** Lưu model vào `results/word2vec_ewt.model`
4. **Demonstrate:** Show similar words và analogies từ model vừa train

**Output mong đợi:**
```
Training Word2Vec model...
Model saved to: results/word2vec_ewt.model

Similar words to 'government':
- administration: 0.8234
- authority: 0.7891
- ...

Analogy: king - man + woman = queen
```

### 2.4. Chạy Advanced Task: Scaling với Apache Spark 

**Yêu cầu:**
- PySpark installed: `pip install pyspark`
- Dataset: `data/c4-train.00000-of-01024-30K.json.gz`

**Chạy Spark job:**
```bash
python test/lab4_spark_word2vec_demo.py
```

**Chức năng:**
- Đọc large JSON dataset (C4 corpus)
- Preprocessing: lowercase, remove punctuation, tokenize
- Train Word2Vec với Spark MLlib (100-dimensional)
- Distributed training trên cluster
- Find synonyms và demonstrate

**Output mong đợi:**
```
Starting Spark session...
Loading C4 dataset...
Preprocessing text...
Training Word2Vec model...
Top 5 words similar to 'computer':
- technology: 0.8234
- software: 0.8012
- ...
```

---

## 3. PHÂN TÍCH KẾT QUẢ

### 3.1. Nhận xét về độ tương đồng và từ đồng nghĩa từ Pre-trained Model

#### **Operation 1: Vector representation của 'king'**

**Kết quả:**
```python
Vector shape: (50,)
Sample values: [0.50451, 0.68607, -0.59517, -0.063669, 0.23106, ...]
```

**Phân tích:**
- Vector có 50 chiều như mong đợi (theo model specification)
- Các giá trị nằm trong khoảng [-1, 1], là normalized vectors
- Mỗi dimension capture một aspect của nghĩa từ "king"
- Không thể interpret từng dimension riêng lẻ (distributed representation)
- Tổng hợp các dimensions mới mang ý nghĩa semantic

#### **Operation 2: Similarity scores**

| Cặp từ | Similarity Score | Phân loại | Nhận xét |
|--------|------------------|-----------|----------|
| king - queen | 0.7839 | Very High | Quan hệ ngữ nghĩa rất gần |
| king - man | 0.5309 | Medium | Có liên quan nhưng khác biệt |

**Phân tích chi tiết:**

1. **'king' và 'queen' (0.7839):**
   
   **Tại sao similarity cao?**
   - Cùng **semantic field**: Hoàng gia, quyền lực, chính trị
   - Cùng **syntactic context**: đều là noun, subject/object positions tương tự
   - Xuất hiện trong **similar contexts**:
     - "The king/queen ruled for 20 years"
     - "king/queen of England"
   - Quan hệ **gender analogy**: king - man + woman ≈ queen
   
   **Ý nghĩa:**
   - Model học được semantic relationship mặc dù không được explicitly trained
   - Word2Vec captures distributional semantics: "You shall know a word by the company it keeps"

2. **'king' và 'man' (0.5309):**
   
   **Tại sao similarity trung bình?**
   - Có **semantic connection**: king is-a man (taxonomic relationship)
   - Nhưng **context khác biệt**:
     - 'king' xuất hiện trong formal, historical contexts
     - 'man' xuất hiện trong everyday, general contexts
   - **Specificity khác nhau**: king (specific role) vs man (general gender)
   
   **Thấp hơn king-queen vì:**
   - Không có symmetrical relationship như king-queen
   - 'man' có nhiều senses và contexts đa dạng hơn
   - Gender relationship không mạnh bằng role relationship

**Kết luận:**
- Model GloVe capture được cả **paradigmatic** (king-queen) và **syntagmatic** (king-man) relationships
- Similarity scores phản ánh chính xác intuition về semantic closeness

#### **Operation 3: 10 từ tương tự với 'computer'**

**Kết quả đầy đủ:**

| Rank | Word | Similarity | Semantic Relation |
|------|------|------------|-------------------|
| 1 | computers | 0.9165 | Plural form (morphological) |
| 2 | software | 0.8815 | Part-of relationship |
| 3 | technology | 0.8526 | Domain category |
| 4 | electronic | 0.8126 | Attribute/property |
| 5 | internet | 0.8060 | Related technology |
| 6 | computing | 0.8026 | Verb form (morphological) |
| 7 | devices | 0.8016 | Hypernym (category) |
| 8 | digital | 0.7992 | Attribute/property |
| 9 | applications | 0.7913 | Use-case relationship |
| 10 | pc | 0.7883 | Synonym/abbreviation |

**Phân tích theo loại semantic relation:**

1. **Morphological variations (hình thái học):**
   - computers (plural), computing (verb form)
   - Model học được morphological relationships tự động
   - Có similarity cao nhất vì share root form

2. **Synonyms/Near-synonyms:**
   - pc (personal computer)
   - Abbreviations được học như separate words

3. **Part-whole relationships:**
   - software (runs on computer)
   - applications (software programs)
   - Meronymy relationship

4. **Attributes/Properties:**
   - electronic, digital
   - Adjectives thường đi với 'computer'
   - Captured through co-occurrence patterns

5. **Domain/Category:**
   - technology (broad category)
   - devices (co-hyponym)
   - Taxonomic relationships

6. **Related concepts:**
   - internet (connected technology)
   - Functional relationships

**Đánh giá chất lượng:**

✅ **Strengths:**
- Không có từ không liên quan (no noise)
- Đa dạng loại relationships
- Top results rất chính xác
- Capture cả syntactic và semantic relations

⚠️ **Observations:**
- Không phân biệt được polysemy (nếu 'computer' có nghĩa khác)
- Bias theo training corpus (Wikipedia + Gigaword)
- Emphasize technical domain terms

### 3.2. Phân tích Document Embedding

#### **Input và Processing:**

**Input document:** "The queen rules the country."

**Tokenization result:**
```
Raw tokens: ['the', 'queen', 'rules', 'the', 'country', '.']
Valid tokens (in vocab): ['the', 'queen', 'rules', 'country']
OOV tokens: ['.']
Coverage: 4/6 tokens (66.7%)
```

#### **Vector Averaging Process:**

```python
# Pseudo-code của quá trình
vectors = [
    model['the'],      # Function word, high frequency
    model['queen'],    # Content word, semantic important
    model['rules'],    # Verb, action
    model['country']   # Noun, semantic important
]

doc_vector = mean(vectors)  # Element-wise average
# Shape: (50,) - same as word vectors
```

#### **Phân tích kết quả:**

**1. Semantic Composition:**

Document vector capture được ý nghĩa tổng thể:
- **Topic:** Monarchy, government, leadership
- **Key concepts:** queen (subject), rules (action), country (object)
- **Semantic frame:** Political power structure

Vector này nên có high similarity với documents về:
- Political systems
- Royal families
- Government structures
- National leadership

**2. Effect of Function Words:**

- 'the' xuất hiện 2 lần → có weight cao hơn trong average
- Function words thường có less specific semantics
- Averaging dilutes thông tin semantic của content words
- **Limitation:** Simple averaging không distinguish important words

**3. Handling OOV:**

- Dấu '.' (punctuation) không trong vocabulary → bị ignore
- **Hợp lý:** Punctuation không contribute semantic meaning
- **Trường hợp khác:** Technical terms, proper nouns có thể là OOV
- **Improvement:** Có thể dùng subword embeddings (FastText)

**4. Vector Properties:**

```python
# Document vector properties
- Magnitude: Smaller than individual words (averaging effect)
- Direction: Points toward semantic center of component words
- Dimension: (50,) - preserves original embedding space
```

#### **Limitations of Simple Averaging:**

1. **No word weighting:**
   - Tất cả words contribute equally
   - Function words vs content words không được distinguish
   - Improvement: TF-IDF weighted averaging

2. **No word order:**
   - "The queen rules the country" = "The country rules the queen"
   - Bag-of-words approach
   - Improvement: Recurrent networks, Transformers

3. **Single vector representation:**
   - Complex documents reduced to one point
   - Loss of fine-grained information
   - Improvement: Hierarchical representations

**Kết luận:**
- Simple averaging là baseline hiệu quả cho short texts
- Phù hợp cho tasks như document similarity, clustering
- Cần sophisticated methods cho longer, complex documents

### 3.3. So sánh Pre-trained vs Self-trained Model (Bonus Task)

#### **Bảng so sánh chi tiết:**

| Tiêu chí | Pre-trained (GloVe) | Self-trained (Word2Vec - UD_EWT) |
|----------|---------------------|-----------------------------------|
| **Training Corpus** | Wikipedia + Gigaword (6B tokens) | UD_English-EWT (~254K tokens) |
| **Vocabulary Size** | 400,000 words | ~5,000 words |
| **Vector Dimension** | 50 | 100 (configurable) |
| **Training Time** | Days on large cluster | Minutes on single machine |
| **Training Algorithm** | GloVe (matrix factorization) | Word2Vec Skip-gram/CBOW |
| **Domain** | General, encyclopedic | Linguistic annotations, varied |
| **OOV Rate** | ~5% for common texts | ~30-40% for general texts |
| **Semantic Quality** | Excellent analogies | Weaker analogies |
| **Syntactic Quality** | Good | Good (trained on parsed data) |
| **Use Cases** | General NLP tasks | Domain-specific, linguistic analysis |

#### **Khi nào dùng Pre-trained:**

✅ **Advantages:**
- Large vocabulary coverage
- Strong semantic relationships
- Good for general domain
- Proven quality
- No training needed

📊 **Best for:**
- Document classification
- Sentiment analysis
- General text understanding
- Prototyping

#### **Khi nào dùng Self-trained:**

✅ **Advantages:**
- Domain-specific vocabulary
- Captures domain jargon
- Adaptable to special needs
- Smaller model size
- No licensing issues

📊 **Best for:**
- Medical, legal, technical domains
- Languages without pre-trained models
- Specialized terminology
- Research purposes

#### **Hybrid Approach:**

**Transfer Learning:**
```python
# Start with pre-trained embeddings
base_model = load_glove('glove-wiki-gigaword-50')

# Fine-tune on domain data
fine_tuned_model = train_word2vec(
    domain_corpus,
    pretrained=base_model
)
```

**Benefits:**
- Combines general knowledge + domain specificity
- Better than pure self-training on small data
- Reduces OOV rate

---

## 4. KHÓ KHĂN VÀ GIẢI PHÁP

### 4.1. Vấn đề tương thích thư viện (Library Compatibility)

**Khó khăn gặp phải:**

```
ERROR: Cannot import name 'triu' from 'scipy.linalg.special_matrices'
ImportError: cannot import name 'downloader' from 'gensim'
```

**Nguyên nhân:**
- Python 3.13.6 là phiên bản rất mới (released 2024)
- gensim 4.x chưa có prebuilt wheel cho Python 3.13 trên Windows
- gensim 0.10.1 (old version) incompatible với scipy 1.16.1
- API changes giữa các versions của scipy
- Build from source requires Visual C++ Build Tools (5-10 GB)

**Giải pháp đã áp dụng:**

1. **Option 1: Use Python 3.11** (Recommended - Đã áp dụng)
   ```bash
   # Python 3.11 có prebuilt wheels cho tất cả dependencies
   # Download Python 3.11 từ python.org
   py -3.11 -m pip install gensim numpy scipy
   ```

2. **Option 2: Install Visual C++ Build Tools**
   - Download từ: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Chọn workload: "Desktop development with C++"
   - Install size: ~6 GB
   - Install time: 15-30 minutes
   - Sau đó: `pip install gensim` sẽ compile from source

3. **Option 3: Use Conda** (Alternative)
   ```bash
   conda install -c conda-forge gensim
   # Conda có binary packages cho nhiều platforms
   ```

**Bài học:**
- Check compatibility matrix trước khi upgrade Python
- Stick with stable, well-supported versions cho production
- Keep environment documentation updated

### 4.2. Memory Management với Large Embeddings

**Khó khăn:**

| Model | File Size | RAM Usage | Download Time |
|-------|-----------|-----------|---------------|
| glove-wiki-gigaword-50 | ~65 MB | ~200 MB | 1-3 mins |
| glove-wiki-gigaword-100 | ~130 MB | ~400 MB | 2-5 mins |
| glove-wiki-gigaword-300 | ~1 GB | ~3 GB | 10-20 mins |

**Issues:**
- Loading 300d model có thể crash trên máy có RAM < 4GB
- First-time download can timeout trên slow connections
- Git repository bloat nếu accidentally commit embeddings
- Multiple projects share cache → disk space waste

**Giải pháp:**

1. **Use smaller dimensions:**
   ```python
   # 50d thường đủ cho most tasks
   embedder = WordEmbedder('glove-wiki-gigaword-50')
   
   # Only use 300d khi cần very high quality
   # embedder = WordEmbedder('glove-wiki-gigaword-300')
   ```

2. **Lazy loading:**
   ```python
   # Model chỉ load khi cần
   # Cache tại ~/.gensim-data/ để reuse
   # Không cần load lại mỗi lần chạy
   ```

3. **Git ignore embeddings:**
   ```gitignore
   # .gitignore
   .embeddings_cache/
   ~/.gensim-data/
   results/*.model
   ```

4. **Document download requirements:**
   ```markdown
   # README.md
   First run will download ~65MB model.
   Ensure stable internet connection.
   ```

**Best practices:**
- Monitor RAM usage: `psutil.virtual_memory()`
- Use memory-mapped files for very large models
- Consider model compression techniques

### 4.3. Handling Out-of-Vocabulary (OOV) Words

**Khó khăn:**

**Common OOV cases:**
- **Domain-specific terms:** "COVID-19", "blockchain", "GPT"
- **Proper nouns:** "Nguyễn", "Hanoi", "ChatGPT"
- **Typos:** "comuter", "quene"
- **Punctuation:** ".", ",", "!"
- **Numbers:** "2024", "3.14"
- **Slang/Informal:** "lol", "omg", "tbh"

**Impact on document embedding:**
```python
# Example with high OOV rate
doc = "The CEO of OpenAI announced GPT-4.5 release in Q1 2024!"
tokens = ['the', 'ceo', 'of', 'openai', 'announced', 'gpt-4.5', 
          'release', 'in', 'q1', '2024', '!']

# In vocab: ['the', 'of', 'announced', 'release', 'in']
# OOV: ['ceo', 'openai', 'gpt-4.5', 'q1', '2024', '!']
# Coverage: 5/11 = 45% (poor!)
```

**Giải pháp:**

1. **Basic handling (Đã implement):**
   ```python
   def get_vector(self, word: str):
       if word in self.model:
           return self.model[word]
       # Option 1: Return None
       return None
       
       # Option 2: Return zero vector
       # return np.zeros(self.model.vector_size)
   ```

2. **Preprocessing để reduce OOV:**
   ```python
   def preprocess(self, word: str):
       # Lowercase
       word = word.lower()
       
       # Remove punctuation
       word = word.strip('.,!?;:')
       
       # Handle contractions
       word = word.replace("'s", "")
       
       return word
   ```

3. **Fallback strategies:**
   ```python
   def get_vector_with_fallback(self, word: str):
       # Try exact match
       if word in self.model:
           return self.model[word]
       
       # Try lowercase
       if word.lower() in self.model:
           return self.model[word.lower()]
       
       # Try without punctuation
       clean = word.strip('.,!?')
       if clean in self.model:
           return self.model[clean]
       
       # Give up
       return None
   ```

4. **Advanced: Character-level embeddings** (Future work)
   ```python
   # Use FastText instead of Word2Vec
   # FastText can generate vectors for OOV words
   # using subword information
   from gensim.models import FastText
   ```

**Metrics to track:**
```python
def compute_coverage(self, tokens):
    in_vocab = sum(1 for t in tokens if t in self.model)
    coverage = in_vocab / len(tokens) if tokens else 0
    return coverage

# Good coverage: > 80%
# Acceptable: 60-80%
# Poor: < 60% (consider different model)
```

### 4.4. Git Repository Management

**Khó khăn:**

**Accidentally committed:**
```
.embeddings_cache/
├── glove.6B.50d.txt          (~171 MB)
├── glove.6B.100d.txt         (~347 MB)
├── glove.6B.200d.txt         (~661 MB)
└── glove.6B.300d.txt         (~990 MB)

results/
└── word2vec_ewt.model        (~12 MB)

data/
└── c4-train...json.gz        (~500 MB)

Total: ~2.6 GB in commits!
```

**Problem:**
- GitHub file limit: 100 MB
- Push rejected with error
- Repository clone very slow
- Wastes GitHub storage quota

**Root cause:**
```bash
# Added .gitignore AFTER committing files
git add .embeddings_cache/    # ❌ BAD
git commit -m "lab4"
# Now files are in history

# Later added .gitignore
echo ".embeddings_cache/" >> .gitignore  # ❌ TOO LATE
```

**Giải pháp đã áp dụng:**

1. **Remove files from git history:**
   ```bash
   # Reset to commit before large files
   git reset --soft aa4ff1d
   
   # Unstage large files
   git restore --staged .embeddings_cache/
   git restore --staged data/*.json.gz
   git restore --staged results/*.model
   
   # Commit only code
   git add src/ test/ reports/ requirements.txt .gitignore
   git commit -m "lab4: Add Word Embeddings implementation"
   
   # Force push (if haven't pushed yet)
   git push origin main
   ```

2. **Proper .gitignore:**
   ```gitignore
   # .gitignore (add BEFORE first commit)
   
   # Embeddings cache
   .embeddings_cache/
   .gensim-data/
   
   # Trained models
   results/*.model
   results/*.bin
   results/*.vec
   
   # Large datasets
   data/*.json.gz
   data/*.zip
   data/*.tar.gz
   
   # Python
   __pycache__/
   *.pyc
   .venv/
   ```

3. **Alternative: Git LFS** (For legitimate large files)
   ```bash
   # Install Git Large File Storage
   git lfs install
   
   # Track large files
   git lfs track "data/*.json.gz"
   git lfs track "results/*.model"
   
   # Add .gitattributes
   git add .gitattributes
   ```

**Prevention checklist:**
- ✅ Create .gitignore FIRST
- ✅ Test with `git status` before commit
- ✅ Use `git add -p` for interactive staging
- ✅ Review `git diff --staged`
- ✅ Keep data separate from code repo

### 4.5. Model Download Timeouts

**Khó khăn:**
```python
# Connection timeout on slow networks
model = api.load('glove-wiki-gigaword-300')
# HTTPError: 504 Gateway Timeout
```

**Giải pháp:**
```python
import gensim.downloader as api

def load_model_with_retry(name, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Loading {name} (attempt {attempt+1}/{max_retries})...")
            model = api.load(name)
            print("Success!")
            return model
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)
            else:
                raise
```

### 4.6. Tokenization Mismatch

**Khó khăn:**
```python
# Model trained với specific tokenization
# "don't" → ["don't"] or ["do", "n't"] or ["don", "'", "t"]?

# SimpleTokenizer: "don't" → ["don't"]
# But model expects: "don't" → ["do", "n't"]
# Result: OOV!
```

**Giải pháp:**
- Use same tokenization as training corpus
- For GloVe: simple whitespace + punctuation split
- Document tokenization requirements
- Test coverage on sample texts

---

## 5. KẾT LUẬN

### 5.1. Tóm tắt những gì đã hoàn thành

#### **Tasks chính (Main Requirements):**

✅ **Task 1: Setup**
- Cài đặt gensim library thành công
- Download và load pre-trained model `glove-wiki-gigaword-50`
- Model tự động cache để reuse
- Hiểu workflow của gensim.downloader API

✅ **Task 2: Word Embedding Exploration**
- Tạo file `src/representations/word_embedder.py`
- Implement class `WordEmbedder` với 4 methods:
  - `__init__(model_name)`: Load model
  - `get_vector(word)`: Get embedding vector
  - `get_similarity(word1, word2)`: Compute cosine similarity
  - `get_most_similar(word, top_n)`: Find similar words
- Handle OOV words properly
- Type hints và documentation đầy đủ

✅ **Task 3: Document Embedding**
- Implement `embed_document(document, tokenizer)`
- Integration với Tokenizer từ Lab 1
- Average pooling strategy
- Handle edge cases (empty doc, all OOV)

✅ **Evaluation:**
- Tạo file `test/lab4_test.py`
- Test đầy đủ 4 operations theo yêu cầu:
  1. Vector cho 'king' ✓
  2. Similarity scores ✓
  3. 10 từ tương tự 'computer' ✓
  4. Document embedding ✓

#### **Bonus Tasks:**

✅ **Bonus: Training Word2Vec from Scratch**
- Script `test/lab4_embedding_training_demo.py`
- Train trên UD_English-EWT corpus
- Memory-efficient streaming với SentenceStreamer
- Save và load trained model
- Demonstrate analogies và similar words

✅ **Advanced: Apache Spark Word2Vec** (Đã code outline)
- Script `test/lab4_spark_word2vec_demo.py`
- PySpark MLlib Word2Vec implementation
- Distributed training workflow
- Scalable cho big data

### 5.2. Kiến thức đã học được

#### **1. Theoretical Understanding:**

**Word Embeddings:**
- Dense vs Sparse representations (so với TF-IDF)
- Distributional semantics: "You shall know a word by the company it keeps"
- Low-dimensional (50-300d) vs high-dimensional (vocab size) representations
- Semantic relationships encoded trong vector space

**Word2Vec:**
- Training approaches: CBOW vs Skip-gram
- Context windows và co-occurrence patterns
- Negative sampling optimization
- Subsampling frequent words

**GloVe:**
- Matrix factorization approach
- Global corpus statistics
- Weighted least squares objective
- Comparison với Word2Vec

#### **2. Practical Skills:**

**Library Usage:**
- gensim API: downloader, KeyedVectors, Word2Vec
- Loading và caching pre-trained models
- Model operations: similarity, most_similar, analogies
- Error handling và OOV management

**Implementation:**
- Clean class design với proper encapsulation
- Method documentation với docstrings
- Type hints cho better code clarity
- Unit testing practices

**Document Representation:**
- Tokenization integration
- Vector averaging strategies
- Weighting schemes (uniform, TF-IDF)
- Alternative approaches (max pooling, hierarchical)

#### **3. Real-world Challenges:**

**Dependency Management:**
- Python version compatibility
- Library version conflicts (gensim + scipy)
- Build tools requirements
- Virtual environments

**Resource Management:**
- Memory constraints với large models
- Disk space cho embeddings cache
- Download bandwidth considerations
- Git repository size management

**Quality Considerations:**
- OOV word coverage
- Domain adaptation needs
- Pre-trained vs custom trade-offs
- Evaluation metrics

### 5.3. Bài học kinh nghiệm (Lessons Learned)

#### **1. Development Best Practices:**

**Environment Management:**
```bash
# ✅ DO: Version pinning
numpy==1.25.2
scipy==1.11.4
gensim==4.3.2

# ❌ DON'T: Latest versions without testing
numpy
scipy
gensim
```

**Git Workflow:**
```bash
# ✅ DO: Proper .gitignore from start
echo ".embeddings_cache/" >> .gitignore
git add .gitignore
git commit -m "Add gitignore"

# ❌ DON'T: Add large files then gitignore
git add .embeddings_cache/  # Already tracked!
echo ".embeddings_cache/" >> .gitignore  # Too late!
```

**Error Handling:**
```python
# ✅ DO: Graceful degradation
try:
    vector = model[word]
except KeyError:
    logger.warning(f"OOV word: {word}")
    vector = None

# ❌ DON'T: Silent failures
vector = model.get(word, None)  # No warning
```

#### **2. Technical Decisions:**

**When to use Pre-trained:**
- ✅ General domain tasks
- ✅ Prototyping phase
- ✅ Limited training data
- ✅ Need proven quality

**When to train Custom:**
- ✅ Domain-specific vocabulary
- ✅ Large domain corpus available (>1M tokens)
- ✅ Specific requirements
- ✅ No suitable pre-trained available

**Model Selection:**
| Size | Use Case | Trade-off |
|------|----------|-----------|
| 50d | Fast prototyping, limited RAM | Lower quality |
| 100d | Good balance | Standard choice |
| 300d | Production, high quality | Memory intensive |

#### **3. Debugging Strategies:**

**OOV Investigation:**
```python
# Compute coverage before processing
tokens = tokenizer.tokenize(doc)
in_vocab = [t for t in tokens if t in model]
coverage = len(in_vocab) / len(tokens)

if coverage < 0.6:
    logger.warning(f"Low coverage: {coverage:.2%}")
    logger.info(f"OOV words: {set(tokens) - set(in_vocab)}")
```

**Performance Profiling:**
```python
import time

start = time.time()
model = api.load('glove-wiki-gigaword-50')
print(f"Load time: {time.time() - start:.2f}s")

start = time.time()
vectors = [model[word] for word in words]
print(f"Lookup time: {time.time() - start:.4f}s")
```

### 5.4. Hướng phát triển tiếp theo (Future Work)

#### **Improvements cho Lab 4:**

1. **Better Document Embeddings:**
   - TF-IDF weighted averaging
   - Use sentence transformers (BERT, Sentence-BERT)
   - Hierarchical representations

2. **Advanced OOV Handling:**
   - FastText subword embeddings
   - Character-level models
   - Morphological analysis

3. **Visualization:**
   - t-SNE plots của word clusters
   - Interactive exploration với TensorBoard
   - Analogy visualization

4. **Evaluation:**
   - Word similarity benchmarks (SimLex-999, WordSim-353)
   - Analogy tasks (Google analogy dataset)
   - Downstream task performance

#### **Extensions:**

1. **Contextual Embeddings:**
   - ELMo (context-dependent)
   - BERT embeddings
   - GPT embeddings

2. **Multilingual:**
   - Cross-lingual embeddings
   - Multilingual BERT
   - Translation invariance

3. **Domain Adaptation:**
   - Fine-tuning strategies
   - Transfer learning
   - Multi-domain models

### 5.5. Kết luận chung

Lab 4 cung cấp foundation vững chắc về word embeddings:

**Về mặt lý thuyết:**
- Hiểu distributed representation
- Semantic relationships trong vector space
- Training algorithms (Word2Vec, GloVe)

**Về mặt thực hành:**
- Sử dụng pre-trained models hiệu quả
- Implement document embeddings
- Handle real-world issues (OOV, memory, compatibility)

**Về kỹ năng:**
- Library usage (gensim)
- Code organization
- Debugging và troubleshooting
- Documentation practices

Word embeddings là core technique trong modern NLP, làm nền tảng cho:
- Sentiment analysis
- Text classification
- Information retrieval
- Machine translation
- Question answering

Moving forward: Contextual embeddings (BERT, GPT) build upon these concepts, adding context-awareness và task-specific fine-tuning.

---

## 6. TRÍCH DẪN TÀI LIỆU

### 6.1. Papers (Nghiên cứu gốc)

1. **Word2Vec:**
   - Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013)
   - "Efficient Estimation of Word Representations in Vector Space"
   - arXiv:1301.3781
   - URL: https://arxiv.org/abs/1301.3781

2. **GloVe:**
   - Pennington, J., Socher, R., & Manning, C. D. (2014)
   - "GloVe: Global Vectors for Word Representation"
   - EMNLP 2014
   - Paper: https://nlp.stanford.edu/pubs/glove.pdf
   - Website: https://nlp.stanford.edu/projects/glove/

3. **Distributed Representations:**
   - Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013)
   - "Distributed Representations of Words and Phrases and their Compositionality"
   - NIPS 2013
   - arXiv:1310.4546

### 6.2. Thư viện và Tools

1. **Gensim**
   - Řehůřek, R., & Sojka, P. (2010)
   - "Software Framework for Topic Modelling with Large Corpora"
   - LREC 2010 Workshop
   - URL: https://radimrehurek.com/gensim/
   - GitHub: https://github.com/RaRe-Technologies/gensim

2. **NumPy**
   - Harris, C. R., et al. (2020)
   - "Array programming with NumPy"
   - Nature 585, 357–362
   - URL: https://numpy.org/

3. **Apache Spark MLlib**
   - Meng, X., et al. (2016)
   - "MLlib: Machine Learning in Apache Spark"
   - Journal of Machine Learning Research 17(34):1-7
   - URL: https://spark.apache.org/mllib/

### 6.3. Datasets

1. **Universal Dependencies - English EWT**
   - Silveira, N., et al. (2014)
   - "A Gold Standard Dependency Corpus for English"
   - LREC 2014
   - URL: https://universaldependencies.org/
   - GitHub: https://github.com/UniversalDependencies/UD_English-EWT

2. **Wikipedia Corpus**
   - Wikimedia Foundation
   - URL: https://dumps.wikimedia.org/

3. **Gigaword Corpus**
   - Graff, D., et al. (2003)
   - English Gigaword Fifth Edition
   - Linguistic Data Consortium

### 6.4. Documentation

1. **Gensim Documentation**
   - Installation: https://radimrehurek.com/gensim/install.html
   - Word2Vec Tutorial: https://radimrehurek.com/gensim/models/word2vec.html
   - Downloader API: https://radimrehurek.com/gensim/downloader.html
   - KeyedVectors: https://radimrehurek.com/gensim/models/keyedvectors.html

2. **NumPy Documentation**
   - User Guide: https://numpy.org/doc/stable/user/index.html
   - Array operations: https://numpy.org/doc/stable/reference/arrays.html
   - Linear algebra: https://numpy.org/doc/stable/reference/routines.linalg.html

3. **PySpark Documentation**
   - MLlib Guide: https://spark.apache.org/docs/latest/ml-guide.html
   - Word2Vec: https://spark.apache.org/docs/latest/ml-features.html#word2vec

4. **Python Documentation**
   - Virtual Environments: https://docs.python.org/3/tutorial/venv.html
   - Type Hints: https://docs.python.org/3/library/typing.html

### 6.5. Tutorials và Courses

1. **Stanford CS224N: Natural Language Processing with Deep Learning**
   - Instructor: Christopher Manning
   - URL: http://web.stanford.edu/class/cs224n/
   - Lectures on Word Vectors (Lectures 1-2)

2. **FastAI NLP Course**
   - URL: https://www.fast.ai/
   - Practical NLP with Deep Learning

3. **Gensim Tutorials**
   - Word2Vec Tutorial: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
   - Doc2Vec Tutorial: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html

### 6.6. Additional Resources

1. **Word Embeddings Explained**
   - Ruder, S. (2016)
   - "On word embeddings - Part 1, 2, 3"
   - Blog: https://ruder.io/word-embeddings-1/

2. **Evaluation Benchmarks**
   - SimLex-999: https://fh295.github.io/simlex.html
   - WordSim-353: http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/
   - Google Analogy Dataset: https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt

3. **Visual C++ Build Tools**
   - Microsoft: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Installation Guide: https://docs.microsoft.com/en-us/cpp/build/

4. **Cosine Similarity**
   - Wikipedia: https://en.wikipedia.org/wiki/Cosine_similarity
   - Mathematical explanation and applications

---

## PHỤ LỤC

### A. Cấu trúc thư mục project

```
d:\NLP\lab\
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── simple_tokenizer.py      # From Lab 1
│   │   └── regex_tokenizer.py       # From Lab 1
│   └── representations/
│       ├── __init__.py
│       └── word_embedder.py          # ⭐ Main Lab 4 implementation
├── test/
│   ├── lab4_test.py                  # ⭐ Evaluation (required)
│   ├── lab4_embedding_training_demo.py  # ⭐ Bonus Task
│   └── lab4_spark_word2vec_demo.py   # ⭐ Advanced Task
├── data/
│   ├── UD_English-EWT/
│   │   └── en_ewt-ud-train.txt      # Training corpus
│   └── c4-train.00000-of-01024-30K.json.gz  # Large corpus for Spark
├── results/
│   └── word2vec_ewt.model           # Trained model (gitignored)
├── reports/
│   └── lab4_report.md               # ⭐ Báo cáo này
├── .embeddings_cache/               # Downloaded embeddings (gitignored)
├── .gitignore                        # ⭐ Important!
├── requirements.txt                  # Dependencies
└── README.md
```

### B. System Requirements

**Minimum:**
- **OS:** Windows 10, Linux (Ubuntu 18.04+), macOS 10.14+
- **Python:** 3.10 or higher
- **RAM:** 2 GB (cho 50d model)
- **Disk:** 1 GB free space
- **Internet:** Stable connection cho first-time download

**Recommended:**
- **Python:** 3.11 (best compatibility)
- **RAM:** 4 GB+ (cho 100d/300d models)
- **Disk:** 5 GB+ (cho multiple models)
- **CPU:** Multi-core (cho training tasks)

**For Bonus Task:**
- **RAM:** 4 GB+
- **Compiler:** Visual C++ Build Tools (Windows) hoặc GCC (Linux)

**For Advanced Task:**
- **RAM:** 8 GB+
- **Disk:** 10 GB+ (Spark + large data)
- **Java:** JDK 8 or 11 (for PySpark)

### C. Dependencies (requirements.txt)

```txt
# Core libraries
numpy>=1.21.0,<2.0.0
scipy>=1.7.0,<2.0.0
gensim>=4.0.0

# Lab 1 dependencies (if using)
# (add your Lab 1 requirements here)

# Optional: For Bonus Task
# (already included with gensim)

# Optional: For Advanced Task
pyspark>=3.0.0

# Development tools (optional)
pytest>=7.0.0
black>=22.0.0
mypy>=0.950
```

### D. Model Specifications

| Model Name | Vocab Size | Vector Dim | File Size | RAM Usage | Download Time* |
|------------|------------|------------|-----------|-----------|----------------|
| glove-wiki-gigaword-50 | 400K | 50 | 65 MB | ~200 MB | 1-3 min |
| glove-wiki-gigaword-100 | 400K | 100 | 130 MB | ~400 MB | 2-5 min |
| glove-wiki-gigaword-200 | 400K | 200 | 252 MB | ~800 MB | 3-8 min |
| glove-wiki-gigaword-300 | 400K | 300 | 376 MB | ~1.2 GB | 5-15 min |
| word2vec-google-news-300 | 3M | 300 | 1.6 GB | ~4 GB | 15-30 min |

*Tuỳ tốc độ internet

### E. Troubleshooting Common Issues

#### **E.1. Import Error: gensim**
```python
ImportError: cannot import name 'downloader' from 'gensim'
```
**Solution:**
```bash
pip install --upgrade gensim
# Ensure gensim >= 4.0.0
```

#### **E.2. Memory Error**
```python
MemoryError: Unable to allocate array
```
**Solution:**
- Use smaller model (50d instead of 300d)
- Close other applications
- Increase system swap/page file

#### **E.3. Download Timeout**
```python
HTTPError: 504 Gateway Timeout
```
**Solution:**
```python
# Retry or download manually
# See section 4.5 in report
```

#### **E.4. Git Push Rejected**
```bash
remote: error: File .embeddings_cache/glove.6B.300d.txt is 990 MB
```
**Solution:**
```bash
# Remove from history (see section 4.4)
git reset --soft <commit_before_large_files>
```

#### **E.5. OOV Rate Too High**
```python
# Coverage < 50%
```
**Solution:**
- Use different/larger model
- Check tokenization compatibility
- Consider FastText for subword handling
- Train custom embeddings on domain corpus


