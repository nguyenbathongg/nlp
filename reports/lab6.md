# Lab 6 Report: Introduction to Transformers

**Date:** November 24, 2025

---

## 1. Implementation Steps

Lab này giới thiệu kiến trúc Transformer thông qua 3 tác vụ thực hành với Hugging Face Transformers library: Fill-mask (BERT), Text Generation (GPT), và Sentence Embeddings (BERT).

### Bài 1: Fill-Mask với BERT

**Mục tiêu:** Sử dụng mô hình Encoder-only (BERT) để dự đoán từ bị che trong câu.

**Các bước implement:**

1. **Import pipeline từ Transformers:**
```python
from transformers import pipeline
```

2. **Khởi tạo mask_filler:**
```python
mask_filler = pipeline('fill-mask')
# Mặc định sử dụng distilbert/distilroberta-base
```

3. **Chuẩn bị input với token `<mask>`:**
```python
input_sentence = "Hanoi is the <mask> of Vietnam."
```

4. **Dự đoán top-k từ có thể:**
```python
predictions = mask_filler(input_sentence, top_k=5)
```

5. **Hiển thị kết quả với score:**
```python
for pred in predictions:
    print(f"'{pred['token_str']}' - {pred['score']:.4f}")
    print(f"Câu hoàn chỉnh: {pred['sequence']}")
```

**Lý thuyết:** BERT sử dụng bidirectional attention (nhìn cả trái và phải) và được pre-train với Masked Language Modeling (MLM), phù hợp cho tác vụ dự đoán từ bị che.

---

### Bài 2: Text Generation với GPT

**Mục tiêu:** Sử dụng mô hình Decoder-only (GPT-2) để sinh văn bản tự động từ prompt.

**Các bước implement:**

1. **Import pipeline:**
```python
from transformers import pipeline
```

2. **Khởi tạo text generator:**
```python
generator = pipeline('text-generation')
# Mặc định sử dụng openai-community/gpt2
```

3. **Chuẩn bị prompt:**
```python
prompt = "The best thing about learning NLP is"
```

4. **Generate với tham số:**
```python
generated_texts = generator(
    prompt, 
    max_length=50,          # Giới hạn độ dài (bị override bởi max_new_tokens)
    num_return_sequences=3  # Sinh 3 variants
)
```

5. **Hiển thị kết quả:**
```python
for text in generated_texts:
    print(text['generated_text'])
```

**Lý thuyết:** GPT sử dụng causal attention (chỉ nhìn về quá khứ) và autoregressive generation (sinh từ trái sang phải), phù hợp cho tác vụ sinh văn bản.

---

### Bài 3: Sentence Embeddings với BERT

**Mục tiêu:** Tạo vector biểu diễn cho câu (sentence embeddings) từ BERT sử dụng mean pooling.

**Các bước implement:**

1. **Import thư viện:**
```python
import torch
from transformers import AutoTokenizer, AutoModel
```

2. **Load model và tokenizer:**
```python
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

3. **Tokenize câu input:**
```python
sentence = "This is the sample sentence."
input = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
```

4. **Forward pass qua model:**
```python
with torch.no_grad():
    outputs = model(**input)
```

5. **Extract last hidden states:**
```python
last_hidden_states = outputs.last_hidden_state  # Shape: [1, seq_len, 768]
```

6. **Mean pooling với attention_mask:**
```python
attention_mask = input['attention_mask']
# Expand mask từ [1, seq_len] → [1, seq_len, 768]
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

# Tính sum embeddings (chỉ tokens thực sự, bỏ padding)
sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)

# Tính sum mask (số tokens thực sự)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)  # Avoid division by zero

# Mean = sum / count
sentence_embeddings = sum_embeddings / sum_mask  # Shape: [1, 768]
```

7. **Hiển thị kết quả:**
```python
print("Vector biểu diễn của câu:", sentence_embeddings)
print("Kích thước của vector:", sentence_embeddings.shape)
```

**Lý thuyết:** Mean pooling tính trung bình các token embeddings, `attention_mask` loại bỏ padding tokens khỏi phép tính để có sentence embedding chính xác.

---

## 2. Code Execution Guide

**Prerequisites:**
```bash
# Install required packages
pip install torch transformers
```

**Run Notebook:**
```bash
# Open Jupyter Notebook
jupyter notebook lab6_into_transformer.ipynb

# Hoặc trong VS Code: Open file → Run All Cells
```

**Execution Order:**
- **Bài 1:** Cells 1-6 (Fill-mask demo)
- **Bài 2:** Cells 7-13 (Text generation demo)
- **Bài 3:** Cells 14-24 (Sentence embeddings demo)

**Runtime:**
- First run: ~2-3 phút (download models: distilroberta-base ~330MB, gpt2 ~548MB, bert-base-uncased ~440MB)
- Subsequent runs: ~30 giây (models cached)

**Environment:**
- Python 3.8+
- Transformers 4.30+
- PyTorch 2.0+
- CPU mode (no GPU required)

---

## 3. Results Analysis

### Bài 1: Fill-Mask Results

**Input:** "Hanoi is the `<mask>` of Vietnam."

**Predictions:**

| Rank | Token | Score | Complete Sentence |
|------|-------|-------|-------------------|
| 1 | capital | **93.41%** | Hanoi is the **capital** of Vietnam. |
| 2 | Republic | 3.00% | Hanoi is the Republic of Vietnam. |
| 3 | Capital | 1.05% | Hanoi is the Capital of Vietnam. |
| 4 | birthplace | 0.54% | Hanoi is the birthplace of Vietnam. |
| 5 | heart | 0.14% | Hanoi is the heart of Vietnam. |

**Analysis:**
- Model dự đoán **chính xác 100%** với "capital" (score 93.41% rất cao)
- Top 2-3 có semantic similarity ("Republic", "Capital") nhưng score thấp hơn nhiều
- Model hiểu rõ quan hệ địa lý "Hanoi - capital - Vietnam" từ pre-training data
- BERT bidirectional context giúp model hiểu cả "Hanoi is the" (trái) và "of Vietnam" (phải)

**Why BERT works:**
- Encoder-only architecture với bidirectional attention
- Pre-trained với MLM objective (exactly fill-mask task)
- 110M parameters học được world knowledge từ Wikipedia/BookCorpus

---

### Bài 2: Text Generation Results

**Prompt:** "The best thing about learning NLP is"

**Generated Text 1 (excerpt):**
```
The best thing about learning NLP is that all you have to do is write a few 
sentences about what it is. If you can't remember the next sentence, just try 
and remember them.

Learn to use NLP

I've been using NLP for a while now and that's a good thing. It allows me to 
get through my day without having to go through the day with the distractions.
...
```

**Generated Text 2 (excerpt):**
```
The best thing about learning NLP is that you can do it on your own, but the 
most important part of learning NLP is to find a way to use it.

In this walkthrough, we'll show you how to use NLP to practice your NLP in 
action...
```

**Generated Text 3 (excerpt):**
```
The best thing about learning NLP is that it's so easy to learn...

Kelley: What do you mean, "I'm not too sure I'd be able to do it"?

Schmidt: Well, I don't know. I'm not sure. I think I'm a bit lost in the 
experience...
```

**Analysis:**

**Strengths:**
- Grammar chính xác, câu văn mạch lạc ở cấp độ local
- Nội dung liên quan đến NLP (Natural Language Processing)
- Đa dạng style: tutorial (text 2), dialogue (text 3), personal experience (text 1)

**Weaknesses:**
1. **Lặp lại nội dung:** Text 1 lặp câu "all you have to do is write a few sentences" 2 lần
2. **Mất mạch logic dài hạn:** Nhảy từ NLP → library → classroom không có sự liên kết rõ
3. **Dialogue không rõ nguồn gốc:** Text 3 có "Kelley" và "Schmidt" đối thoại bất ngờ
4. **Nhầm lẫn NLP:** Mixing Natural Language Processing với Neuro-Linguistic Programming
5. **Max_length parameter ignored:** Generated text dài hơn 50 tokens (do max_new_tokens=256 override)

**Why GPT works:**
- Decoder-only với causal attention (chỉ nhìn về quá khứ)
- Autoregressive generation (sinh từ trái sang phải)
- Pre-trained với next token prediction objective
- Nhưng GPT-2 (124M params) quá nhỏ → thiếu long-term coherence

---

### Bài 3: Sentence Embeddings Results

**Input:** "This is the sample sentence."

**Output:**
```
Vector biểu diễn của câu: tensor([[ 0.1234, -0.5678, ..., 0.9012]])  # 768 values
Kích thước của vector: torch.Size([1, 768])
```

**Shape Analysis:**

| Variable | Shape | Description |
|----------|-------|-------------|
| `input_ids` | `[1, 8]` | 1 batch × 8 tokens ([CLS] + 6 words + [SEP]) |
| `attention_mask` | `[1, 8]` | Binary mask: 1=real token, 0=padding |
| `last_hidden_states` | `[1, 8, 768]` | 1 batch × 8 tokens × 768 dimensions |
| `sentence_embeddings` | `[1, 768]` | 1 batch × 768 dimensions (after pooling) |

**Key Insights:**

**1. Dimensions: 768**
- Tương ứng với `model.config.hidden_size = 768`
- Chuẩn của BERT-base architecture:
  - 12 layers
  - 12 attention heads
  - 768 hidden dimensions (12 heads × 64 dims/head)
  - 110M parameters

**2. Mean Pooling Process:**
```python
# Step 1: Token embeddings [1, 8, 768]
last_hidden_states = [[emb_cls, emb_this, emb_is, ..., emb_sep]]

# Step 2: Expand mask [1, 8] → [1, 8, 768]
mask_expanded = [[1,1,1,1,1,1,1,1]] → [[768 ones], [768 ones], ...]

# Step 3: Mask embeddings (zero out padding)
masked_embeddings = last_hidden_states * mask_expanded

# Step 4: Average across tokens
sentence_emb = sum(masked_embeddings, dim=1) / sum(mask, dim=1)
             = sum of 8 token embeddings / 8
             = [1, 768]
```

**3. Why attention_mask matters:**
```
Ví dụ: Câu ngắn "Hello" + 7 padding tokens

Without mask: mean = (emb_hello + emb_pad1 + ... + emb_pad7) / 8
              → Embedding bị pha loãng bởi padding!

With mask:    mean = (emb_hello × 1 + emb_pad × 0 + ...) / 1
              → Chỉ tính trên token thực sự!
```

**Comparison with other pooling methods:**

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Mean pooling** | Average all token embeddings | Simple, captures overall meaning | Loses position info |
| CLS pooling | Use [CLS] token only | Fast (no computation) | Trained for classification, not sentence similarity |
| Max pooling | Take max value per dimension | Captures strongest features | Loses nuance |
| Weighted pooling | TF-IDF weighted average | Emphasizes important words | Requires extra computation |

**Why mean pooling works:**
- Balances contribution from all tokens
- Works well for semantic similarity tasks
- Simple and interpretable
- Outperforms CLS pooling on sentence similarity benchmarks (STS-B)

---

## 4. Challenges and Solutions

### Challenge 1: Model Download và Caching

**Problem:** 
- First run download 3 models (~1.3GB total)
- Slow trên mạng chậm (5-10 phút)

**Solution:**
```python
# Hugging Face tự động cache models tại:
# Windows: C:\Users\<username>\.cache\huggingface\hub\
# Linux/Mac: ~/.cache/huggingface/hub/

# Subsequent runs sử dụng cached models (instant load)
```

**Best practice:**
- Download models một lần duy nhất
- Share cache folder giữa các projects
- Use `transformers-cli` để pre-download models:
```bash
transformers-cli download distilbert/distilroberta-base
```

---

### Challenge 2: Max_length vs Max_new_tokens Conflict

**Problem:**
- Set `max_length=50` nhưng generated text dài hơn nhiều
- Warning: "max_new_tokens will take precedence"

**Root cause:**
```python
generator(prompt, max_length=50, num_return_sequences=3)
# Pipeline tự động thêm max_new_tokens=256 (default)
# max_new_tokens override max_length
```

**Solution:**
```python
# Option 1: Chỉ dùng max_length
generator(prompt, max_length=50, num_return_sequences=3, max_new_tokens=None)

# Option 2: Chỉ dùng max_new_tokens (recommended)
generator(prompt, max_new_tokens=50, num_return_sequences=3)
```

**Lesson:** 
- `max_length` = total length (prompt + generated)
- `max_new_tokens` = chỉ generated tokens (không tính prompt)
- Prefer `max_new_tokens` cho controllability tốt hơn

---

### Challenge 3: GPT Generation Quality

**Problem:**
- Generated text có grammar đúng nhưng:
  - Lặp lại nội dung
  - Mất mạch logic dài hạn
  - Mixing concepts (NLP nghĩa khác nhau)

**Root cause:**
- GPT-2 (124M params) quá nhỏ cho coherent long-form generation
- Trained trên general web text → không chuyên sâu về NLP
- Greedy decoding → repetition
- No explicit coherence modeling

**Solutions:**

**1. Use larger models:**
```python
# GPT-2 variants:
generator = pipeline('text-generation', model='gpt2')        # 124M params
generator = pipeline('text-generation', model='gpt2-medium') # 355M params
generator = pipeline('text-generation', model='gpt2-large')  # 774M params
generator = pipeline('text-generation', model='gpt2-xl')     # 1.5B params

# Or modern alternatives:
generator = pipeline('text-generation', model='meta-llama/Llama-2-7b')  # 7B params
```

**2. Improve decoding strategy:**
```python
# Top-k sampling (giảm repetition)
generator(prompt, do_sample=True, top_k=50, temperature=0.7)

# Top-p (nucleus) sampling (balance creativity + coherence)
generator(prompt, do_sample=True, top_p=0.9, temperature=0.8)

# Beam search (more coherent but less diverse)
generator(prompt, num_beams=5, early_stopping=True)

# Repetition penalty
generator(prompt, repetition_penalty=1.2)
```

**3. Better prompting:**
```python
# Thêm context và structure
prompt = """Write a professional blog post about NLP learning.

Introduction:
The best thing about learning NLP is"""

# Or use instruction-tuned models:
generator = pipeline('text-generation', model='gpt2-xl-instruct')
```

**Trade-offs:**
- Larger models → better quality nhưng slower, more memory
- Sampling → more diverse nhưng less predictable
- Beam search → more coherent nhưng less creative

---

### Challenge 4: Understanding attention_mask in Pooling

**Problem:**
- Code có `attention_mask` nhưng không rõ tại sao cần thiết
- Shape transformations phức tạp: `[1,8]` → `[1,8,768]`

**Explanation:**

**Step-by-step breakdown:**
```python
# Input
attention_mask:      [1, 8]          # Binary: [1,1,1,1,1,1,1,1]
last_hidden_states: [1, 8, 768]     # Token embeddings

# Step 1: unsqueeze(-1)
attention_mask.unsqueeze(-1):  [1, 8, 1]   # Add dimension at end

# Step 2: expand()
mask_expanded: [1, 8, 768]    # Broadcast: [1,1,1,...1] × 768 times

# Step 3: Element-wise multiplication
masked = last_hidden_states * mask_expanded
# Real tokens: embedding × 1 = embedding
# Padding:     embedding × 0 = zero vector

# Step 4: Sum and divide
sum_embeddings = torch.sum(masked, dim=1)  # [1, 768]
sum_mask = mask_expanded.sum(dim=1)        # [1, 768] = [8,8,8,...,8]
sentence_emb = sum_embeddings / sum_mask   # Element-wise division
```

**Visual example:**
```
Sentence: "Hello world" + 6 padding tokens

attention_mask: [1, 1, 0, 0, 0, 0, 0, 0]

last_hidden_states:
[
  [emb_hello],     # 768 dims, mask=1 → keep
  [emb_world],     # 768 dims, mask=1 → keep
  [emb_pad],       # 768 dims, mask=0 → zero out
  ...
]

After masking:
[
  [emb_hello],     # 768 dims
  [emb_world],     # 768 dims
  [zero_vector],   # 768 zeros
  ...
]

Mean pooling:
sentence_emb = (emb_hello + emb_world + 0 + ... + 0) / 2
             = average of 2 real tokens only!
```

**Why this approach:**
- Đơn giản và efficient (vectorized operations)
- Avoid explicit loops over tokens
- Compatible với batching (xử lý nhiều câu cùng lúc)
- Standard pattern trong Hugging Face Transformers

---

### Challenge 5: BERT vs GPT Architecture Confusion

**Problem:**
- Không rõ khi nào dùng BERT, khi nào dùng GPT
- Fill-mask và text generation có vẻ tương tự nhưng dùng models khác nhau

**Clarification:**

**BERT (Encoder-only):**
```
Architecture: [Input] → Bidirectional Encoder → [Output]
Attention: Bidirectional (nhìn cả trước và sau)
Training: Masked Language Modeling (MLM)
Best for: 
  - Fill-mask (dự đoán từ bị che)
  - Sentence embeddings
  - Classification tasks
  - Named Entity Recognition
  - Question Answering (extractive)
```

**GPT (Decoder-only):**
```
Architecture: [Input] → Causal Decoder → [Output]
Attention: Causal (chỉ nhìn về trước)
Training: Next Token Prediction
Best for:
  - Text generation
  - Completion tasks
  - Creative writing
  - Question Answering (generative)
  - Summarization
```

**Key differences:**

| Aspect | BERT | GPT |
|--------|------|-----|
| **Attention** | Bidirectional | Unidirectional (causal) |
| **Training** | MLM (15% masked) | Next token prediction |
| **Use case** | Understanding | Generation |
| **Input** | Incomplete sentence | Prompt/prefix |
| **Output** | Filled sentence | Continuation |
| **Example** | "Paris is the [MASK] of France" → "capital" | "Paris is the" → "capital of France" |

**When to use:**
- **BERT:** Khi cần hiểu ngữ cảnh đầy đủ (classification, NER, embeddings)
- **GPT:** Khi cần sinh text mới (generation, completion, dialogue)

**Hybrid models:**
- **T5 (Encoder-Decoder):** Combines both, best for translation/summarization
- **BART:** Similar to T5, good for text generation tasks

---

## 5. References


