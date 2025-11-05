# Naive English→French Machine Translation (Word Embeddings)

A compact, education-focused project that demonstrates how **word embeddings** can be used to build a **naive, word-level English→French translation system**—without training a large neural model. The approach relies on **cosine similarity**, **nearest neighbors**, and optional **embedding alignment** to map English words to their closest French equivalents in vector space.

> 📌 This repository is ideal as a learning artifact for vector space semantics, basic MT ideas, and practical NLP prototyping in Python/Jupyter.

---

## 🚀 What This Project Does

- Loads **pre-computed word embeddings** (English & French).
- Uses **cosine similarity** + **nearest neighbors** to translate words.
- (Optional) Learns a **linear alignment** between the two embedding spaces.
- Evaluates translations with a small **bilingual test dictionary**.
- Visualizes similarity and provides sanity checks.
- All steps are contained in a single Jupyter notebook: `MachineTranslationnb.ipynb`.

---

## 🔍 Method Overview

1. **Embeddings**: Load pre-trained embeddings (e.g., `.txt`, `.vec`, or `.npy` formats).  
2. **Nearest-Neighbor Translation**: For an English word vector \( v_{en} \), find the French word whose vector has **maximum cosine similarity**.  
3. **Embedding Alignment (Optional)**: Learn a linear map \( W \) that aligns English vectors to the French space by minimizing \( \| W \cdot v_{en} - v_{fr} \|^2 \) over a seed dictionary.  
4. **Evaluation**: Compare predicted translations against a bilingual gold list; report accuracy/top-k accuracy.  
5. **Error Analysis**: Inspect failures (e.g., morphology, polysemy, OOVs).

---

## 🧰 Requirements

- Python 3.9+
- Jupyter
- `numpy`, `pandas`
- `scipy`, `scikit-learn`
- `matplotlib` (optional, for plots)

Install quickly:

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter
```

---

## 📂 Repository Structure

```
.
├── MachineTranslationnb.ipynb        # Main notebook
├── data/
│   ├── en.vec                        # (example) English embeddings
│   ├── fr.vec                        # (example) French embeddings
│   ├── bilingual_dictionary.tsv      # (example) seed dictionary for alignment/eval
│   └── stopwords.txt                 # (optional)
└── README.md
```

> ℹ️ Use whatever embedding files you have. If not included, place your own under `data/` and update the notebook paths.

---

## ▶️ How to Run

1. Clone this repository and open the notebook:
   ```bash
   git clone <your-repo-url>.git
   cd <repo-name>
   jupyter notebook MachineTranslationnb.ipynb
   ```

2. Edit the **paths** in the **data loading** cell to point to your embedding files.

3. Run all cells to:
   - Load embeddings  
   - (Optionally) learn alignment \(W\)  
   - Translate a list of English words  
   - Evaluate on the test dictionary

---

## ✨ Example Usage (Pseudocode)

```python
# Load embeddings
en = load_embeddings("data/en.vec")     # dict: word -> vector
fr = load_embeddings("data/fr.vec")

# Optional: learn alignment using a seed dictionary
seed = load_bilingual_dictionary("data/bilingual_dictionary.tsv")
W = learn_alignment(en, fr, seed)       # linear map

# Translate (with or without alignment)
def translate_word(w):
    v = en[w]
    v_map = W @ v if W is not None else v
    return nearest_neighbor(v_map, fr)  # returns best French candidate

translate_word("cat")   # -> "chat"
```

---

## 🧪 Evaluation

- **Metric**: top-1 accuracy on a bilingual test list.  
- **Extensions**: report **top-k** (k=5) accuracy and show nearest-neighbor lists to analyze close confusions.  
- **Notes**: Word-level systems are sensitive to morphology (pluralization, conjugation) and domain mismatch between embeddings.

---

## 📈 Results (Example Template)

| Setting                         | Top-1 Acc. | Top-5 Acc. |
|---------------------------------|------------|------------|
| Raw embeddings (no alignment)   |  —         | —          |
| With linear alignment (W)       |  —         | —          |

> Replace dashes with your observed results from the notebook.

---

## 🔬 Limitations & Future Work

- Works at the **word level** only; no syntax or reordering.  
- Struggles with **OOV words**, morphology, and polysemy.  
- Improve with: subword embeddings (fastText), phrase tables, alignment regularization, or a small **seq2seq** baseline for comparison.

---

## 📎 Citing Resources / Inspiration

- Mikolov et al. (2013): Word2Vec & vector arithmetic  
- Smith et al. (2017): Offline bilingual word vectors & Procrustes alignment  
- fastText embeddings for multilingual experiments

*(If you used any specific datasets or tutorials, cite them here.)*

---

## 📜 License

MIT License (feel free to change).

---

## 🙏 Acknowledgments

Thanks to open-source NLP libraries and embedding providers.  
Special thanks to courses & materials that inspired this project (e.g., **NLP with Classification & Vector Spaces**, Coursera).
