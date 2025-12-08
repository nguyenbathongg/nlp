# Lab 6 Report: Dependency Parsing với spaCy

**Date:** December 8, 2025

---

## 1. Implementation Steps

Lab này thực hành phân tích cú pháp phụ thuộc (Dependency Parsing) sử dụng thư viện spaCy, bao gồm 5 phần: Load model, Visualize cây phụ thuộc, Trích xuất thuộc tính, Phân tích ngữ nghĩa, và các bài tập nâng cao.

### Phần 2: Load Model và Visualize

**Mục tiêu:** Sử dụng spaCy để parse câu và visualize cây dependency.

```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_md")
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)
displacy.serve(doc, style="dep")
```

**Phân tích cây dependency:**
- **ROOT:** `jumps` - động từ chính của câu
- **Dependents của jumps:** 
  - `fox` (nsubj): chủ ngữ
  - `over` (prep): giới từ bổ nghĩa
- **Head của fox:** `The` (det), `quick` (amod), `brown` (amod)

---

### Phần 3: Trích xuất Dependency Attributes

**Mục tiêu:** Trích xuất các thuộc tính token: TEXT, DEP, HEAD TEXT, HEAD POS, CHILDREN

```python
text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:<12} | {token.dep_:<10} | {token.head.text:<12} | {token.head.pos_:<8} | {children}")
```

**Các thuộc tính quan trọng:**
- `token.dep_`: Quan hệ dependency (nsubj, dobj, prep, ROOT, ...)
- `token.head`: Token cha trong cây
- `token.children`: Các token con
- `token.pos_`: Part-of-speech tag

---

### Phần 4: Trích xuất Subject-Verb-Object Triplets

**Bài 4.1: Tìm triplet (Subject, Verb, Object)**

```python
text = "The cat chased the mouse and the dog watched them."
doc = nlp(text)

for token in doc:
    if token.pos_ == "VERB":
        verb = token.text
        subject, obj = "", ""
        for child in token.children:
            if child.dep_ == "nsubj":
                subject = child.text
            if child.dep_ == "dobj":
                obj = child.text
        if subject and obj:
            print(f"Found Triplet: ({subject}, {verb}, {obj})")
```

**Bài 4.2: Tìm tính từ bổ nghĩa cho danh từ**

```python
text = "The big, fluffy white cat is sleeping on the warm mat."
doc = nlp(text)

for token in doc:
    if token.pos_ == "NOUN":
        adjectives = [child.text for child in token.children if child.dep_ == "amod"]
        if adjectives:
            print(f"Danh từ '{token.text}' được bổ nghĩa bởi các tính từ: {adjectives}")
```

---

### Phần 5: Bài tập nâng cao

**Bài 1: Tìm động từ chính (ROOT)**

```python
def find_main_verb(doc):
    """Tìm động từ chính của câu (token có quan hệ ROOT)"""
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None
```

**Bài 2: Trích xuất Noun Chunks**

```python
def get_noun_chunks(doc):
    """Trích xuất cụm danh từ gồm danh từ + các từ bổ nghĩa (det, amod, compound)"""
    noun_chunks = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            chunk_tokens = [token]
            for t in doc:
                if t.head == token and t.dep_ in ["det", "amod", "compound", "poss", "nummod"]:
                    chunk_tokens.append(t)
            chunk_tokens.sort(key=lambda x: x.i)
            chunk_text = " ".join([t.text for t in chunk_tokens])
            if chunk_text not in noun_chunks:
                noun_chunks.append(chunk_text)
    return noun_chunks
```

**Bài 3: Tìm đường đi đến ROOT**

```python
def get_path_to_root(token):
    """Tìm đường đi từ token lên đến ROOT"""
    path = [token]
    current = token
    while current.head != current:  # ROOT có đặc điểm: head == chính nó
        current = current.head
        path.append(current)
    return path

def get_path_between_tokens(token1, token2):
    """Tìm đường đi ngắn nhất giữa hai token qua LCA (Lowest Common Ancestor)"""
    path1 = get_path_to_root(token1)
    path2 = get_path_to_root(token2)
    
    # Tìm LCA
    path1_set = set(path1)
    for i, t in enumerate(path2):
        if t in path1_set:
            lca = t
            lca_idx = i
            break
    
    # Xây dựng đường đi
    path_to_lca = path1[:path1.index(lca)]
    path_from_lca = path2[:lca_idx + 1][::-1]
    return path_to_lca + path_from_lca
```

---

## 2. Code Execution Guide

**Prerequisites:**
```bash
pip install spacy
python -m spacy download en_core_web_md
```

**Run Notebook:**
```bash
jupyter notebook lab6_dependency_parsing_pandoc.ipynb
# Hoặc trong VS Code: Open file → Run All Cells
```

**Execution Order:**
- Phần 2: Cells 1-6 (Load model, visualize)
- Phần 3: Cells 7-10 (Extract attributes)
- Phần 4: Cells 11-13 (Triplet extraction)
- Phần 5: Cells 14-17 (Advanced exercises)

**Runtime:** ~5 giây (model đã được cache)

---

## 3. Results Analysis

### Phần 3: Dependency Attributes Output

**Input:** "Apple is looking at buying U.K. startup for $1 billion"

| Token | DEP | HEAD | HEAD POS | Children |
|-------|-----|------|----------|----------|
| Apple | nsubj | looking | VERB | [] |
| is | aux | looking | VERB | [] |
| looking | **ROOT** | looking | VERB | [Apple, is, at] |
| at | prep | looking | VERB | [buying] |
| buying | pcomp | at | ADP | [startup] |
| U.K. | compound | startup | NOUN | [] |
| startup | dobj | buying | VERB | [U.K., for] |
| for | prep | startup | NOUN | [billion] |
| $ | quantmod | billion | NUM | [] |
| 1 | compound | billion | NUM | [] |
| billion | pobj | for | ADP | [$, 1] |

**Phân tích:**
- ROOT: `looking` là động từ chính (progressive tense: is looking)
- Cấu trúc: Apple (S) + is looking (V) + at buying startup (complement)
- Noun phrase: "U.K. startup" với compound modifier

---

### Phần 4: Triplet Extraction Results

**Input:** "The cat chased the mouse and the dog watched them."

**Output:**
```
Found Triplet: (cat, chased, mouse)
Found Triplet: (dog, watched, them)
```

**Input:** "The big, fluffy white cat is sleeping on the warm mat."

**Output:**
```
Danh từ 'cat' được bổ nghĩa bởi các tính từ: ['big', 'fluffy', 'white']
Danh từ 'mat' được bổ nghĩa bởi các tính từ: ['warm']
```

---

### Phần 5: Advanced Exercises Results

**Bài 1: Find Main Verb**

| Câu | Main Verb (ROOT) |
|-----|------------------|
| The quick brown fox jumps over the lazy dog. | **jumps** |
| Apple is looking at buying U.K. startup | **looking** |
| The cat chased the mouse and the dog watched them. | **chased** |
| She has been studying for the exam all day. | **studying** |

---

**Bài 2: Noun Chunks Comparison**

| Câu | spaCy noun_chunks | Hàm tự viết |
|-----|-------------------|-------------|
| The quick brown fox jumps over the lazy dog. | ['The quick brown fox', 'the lazy dog'] | ['The quick brown fox', 'the lazy dog'] ✓ |
| The big, fluffy white cat is sleeping | ['The big, fluffy white cat', 'the warm mat'] | ['The big fluffy white cat', 'the warm mat'] |
| Apple Inc. announced a new iPhone model | ['Apple Inc.', 'a new iPhone model'] | ['Apple', 'Apple Inc.', 'iPhone', 'a new iPhone model', 'yesterday'] |

**Nhận xét:**
- Câu đơn giản: Kết quả khớp 100%
- Câu phức tạp: Hàm tự viết có thể trả về nhiều chunk hơn (do không loại bỏ sub-chunks)
- spaCy sử dụng heuristics phức tạp hơn để xác định ranh giới noun chunk

---

**Bài 3: Path to ROOT**

**Input:** "The quick brown fox jumps over the lazy dog."

| Token | Path to ROOT |
|-------|--------------|
| The | The(det) → fox(nsubj) → jumps(ROOT) |
| quick | quick(amod) → fox(nsubj) → jumps(ROOT) |
| brown | brown(amod) → fox(nsubj) → jumps(ROOT) |
| fox | fox(nsubj) → jumps(ROOT) |
| jumps | jumps(ROOT) |
| over | over(prep) → jumps(ROOT) |
| the | the(det) → dog(pobj) → over(prep) → jumps(ROOT) |
| lazy | lazy(amod) → dog(pobj) → over(prep) → jumps(ROOT) |
| dog | dog(pobj) → over(prep) → jumps(ROOT) |

**Đường đi giữa hai token:**
- `The` ↔ `dog`: The → fox → jumps → over → dog (5 nodes)
- `quick` ↔ `lazy`: quick → fox → jumps → over → dog → lazy (6 nodes)
- `fox` ↔ `jumps`: fox → jumps (2 nodes)

---

## 4. Challenges and Solutions

### Challenge 1: Hiểu cấu trúc cây Dependency

**Problem:** Ban đầu khó hình dung cây dependency và các quan hệ head-child.

**Solution:**
- Sử dụng `displacy.serve()` để visualize cây
- Nhận ra: ROOT là gốc, mỗi token có đúng 1 head, có thể có nhiều children
- Đặc điểm ROOT: `token.head == token`

---

### Challenge 2: Noun Chunks không khớp với spaCy

**Problem:** Hàm tự viết trả về nhiều chunks hơn spaCy.

**Root cause:**
- spaCy dùng heuristics để loại bỏ sub-chunks (ví dụ: không trả về "Apple" khi đã có "Apple Inc.")
- Hàm tự viết duyệt tất cả NOUN/PROPN → tạo chunks từ mỗi danh từ

**Solution:**
```python
# Thêm logic loại bỏ sub-chunks
seen_indices = set()
for token in doc:
    if token.i in seen_indices:
        continue  # Bỏ qua token đã thuộc chunk khác
    # ... tạo chunk và đánh dấu các token đã dùng
```

---


## 5. References
