"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


# ---------------------------------------------------------------------------
# Task 1 – TF-IDF
# ---------------------------------------------------------------------------

def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    return sklearn_cosine(tfidf_matrix)


# ---------------------------------------------------------------------------
# Task 2 – GloVe
# ---------------------------------------------------------------------------

def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]
    if not vectors:
        return np.zeros(50, dtype=np.float32)
    return np.mean(vectors, axis=0)


# ---------------------------------------------------------------------------
# Task 3 – DistilBERT
# ---------------------------------------------------------------------------

def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Applies mean pooling over non-padding tokens.
    Returns a numpy array of shape (768,).
    """
    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden    = outputs.last_hidden_state               # (1, seq_len, 768)
    attention_mask = inputs["attention_mask"]                # (1, seq_len)
    mask_expanded  = attention_mask.unsqueeze(-1).float()    # (1, seq_len, 1)

    sum_hidden  = (last_hidden * mask_expanded).sum(dim=1)   # (1, 768)
    sum_mask    = mask_expanded.sum(dim=1).clamp(min=1e-9)   # (1, 1)
    mean_pooled = (sum_hidden / sum_mask).squeeze(0)         # (768,)

    return mean_pooled.numpy()


def build_bert_matrix(texts, tokenizer, model):
    """Pre-compute BERT embeddings for all texts in the corpus.

    Prints a live progress counter so you can see it working.
    Returns a numpy array of shape (n, 768).
    """
    embeddings = []
    total = len(texts)
    for i, text in enumerate(texts):
        embeddings.append(extract_bert_embedding(text, tokenizer, model))
        print(f"\r  BERT corpus embeddings: {i + 1}/{total}", end="", flush=True)
    print()  # newline after progress line
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Task 4 – Compare Similarity Rankings
# ---------------------------------------------------------------------------

def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer,
                         bert_corpus_matrix=None):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself. Return:

        {query_text: {"tfidf": [(text, score), ...],
                      "glove": [(text, score), ...],
                      "bert":  [(text, score), ...]}}

    Parameters
    ----------
    bert_corpus_matrix : np.ndarray or None
        Pre-computed BERT embeddings for *texts* (shape n x 768).
        Pass the result of build_bert_matrix() to avoid recomputing
        embeddings for every query. If None the matrix is computed here
        (slow — one forward-pass per text per query).
    """
    # ── Build GloVe corpus matrix once ──────────────────────────────────────
    print("  Building GloVe corpus matrix...", flush=True)
    glove_matrix = np.array(
        [text_to_glove(t, glove_embeddings) for t in texts],
        dtype=np.float32,
    )
    print(f"  GloVe corpus matrix ready: {glove_matrix.shape}")

    # ── Use or build BERT corpus matrix ─────────────────────────────────────
    if bert_corpus_matrix is None:
        print("  No pre-computed BERT matrix supplied — computing now (slow)...")
        bert_corpus_matrix = build_bert_matrix(texts, bert_tokenizer, bert_model)
    print(f"  BERT corpus matrix ready:  {bert_corpus_matrix.shape}")

    # ── Per-query comparisons ────────────────────────────────────────────────
    results = {}

    for q_num, query in enumerate(queries, 1):
        print(f"\n  [Query {q_num}/{len(queries)}] {query[:80]}...")

        try:
            q_idx = texts.index(query)
        except ValueError:
            q_idx = None

        entry = {}

        # TF-IDF
        if q_idx is not None:
            scores = tfidf_sim[q_idx].copy()
            scores[q_idx] = -1.0
        else:
            scores = np.zeros(len(texts))

        top_idx = np.argsort(scores)[::-1][:3]
        entry["tfidf"] = [(texts[i], float(scores[i])) for i in top_idx]
        print(f"    TF-IDF  top-1: {texts[top_idx[0]][:70]}...")

        # GloVe
        q_glove      = text_to_glove(query, glove_embeddings).reshape(1, -1)
        glove_scores = sklearn_cosine(q_glove, glove_matrix).flatten()

        if q_idx is not None:
            glove_scores[q_idx] = -1.0

        top_idx = np.argsort(glove_scores)[::-1][:3]
        entry["glove"] = [(texts[i], float(glove_scores[i])) for i in top_idx]
        print(f"    GloVe   top-1: {texts[top_idx[0]][:70]}...")

        # BERT
        q_bert      = extract_bert_embedding(query, bert_tokenizer, bert_model)
        bert_scores = sklearn_cosine(
            q_bert.reshape(1, -1), bert_corpus_matrix
        ).flatten()

        if q_idx is not None:
            bert_scores[q_idx] = -1.0

        top_idx = np.argsort(bert_scores)[::-1][:3]
        entry["bert"] = [(texts[i], float(bert_scores[i])) for i in top_idx]
        print(f"    BERT    top-1: {texts[top_idx[0]][:70]}...")

        results[query] = entry

    return results


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModel

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # ── Task 1: TF-IDF ────────────────────────────────────────────────────
    print("\n[Task 1] Building TF-IDF representations...")
    tfidf_matrix, vectorizer = build_tfidf(texts)
    print(f"  TF-IDF matrix shape:            {tfidf_matrix.shape}")
    tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
    print(f"  TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # ── Task 2: GloVe ─────────────────────────────────────────────────────
    print("\n[Task 2] Loading GloVe vectors...")
    glove = load_glove("data/glove_50k_50d.txt")
    print(f"  Loaded {len(glove)} GloVe vectors")
    sample_emb = text_to_glove(texts[0], glove)
    print(f"  Sample GloVe embedding shape: {sample_emb.shape}")

    # OOV rate (Task 5 helper — printed now for convenience)
    all_words = [w.lower() for t in texts for w in t.split()]
    oov_words = [w for w in all_words if w not in glove]
    oov_rate  = len(oov_words) / len(all_words) if all_words else 0.0
    print(f"  OOV rate: {oov_rate:.2%}  ({len(set(oov_words))} unique OOV types)")

    # ── Task 3: DistilBERT ────────────────────────────────────────────────
    print("\n[Task 3] Loading DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model     = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    print(f"  Sample BERT embedding shape: {sample_bert.shape}")

    # Pre-compute BERT embeddings for the ENTIRE corpus once.
    # compare_similarities reuses this matrix for every query.
    print("\n  Pre-computing BERT embeddings for full corpus (one-time cost)...")
    bert_matrix = build_bert_matrix(texts, tokenizer, model)
    print(f"  BERT corpus matrix shape: {bert_matrix.shape}")

    # ── Task 4: Compare ───────────────────────────────────────────────────
    print("\n[Task 4] Running per-query similarity comparison...")
    queries = [df[df["category"] == cat]["text"].iloc[0]
               for cat in df["category"].unique()]

    comparison = compare_similarities(
        texts, queries, tfidf_sim, glove, model, tokenizer,
        bert_corpus_matrix=bert_matrix,   # pre-computed; no redundant passes
    )

    # ── Final table ───────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SIMILARITY COMPARISON — top-3 results per method")
    print("=" * 100)

    for q in comparison:
        print(f"\n{'─'*100}")
        print(f"QUERY: {q[:120]}...")
        print(f"{'─'*100}")
        print(f"  {'RANK':<5} {'TF-IDF (score)':<46} {'GloVe (score)':<46} {'BERT (score)':<46}")
        print(f"  {'----':<5} {'-'*44:<46} {'-'*44:<46} {'-'*44:<46}")

        tfidf_top = comparison[q]["tfidf"]
        glove_top = comparison[q]["glove"]
        bert_top  = comparison[q]["bert"]

        for rank in range(3):
            t_txt, t_sc = tfidf_top[rank] if rank < len(tfidf_top) else ("—", 0.0)
            g_txt, g_sc = glove_top[rank] if rank < len(glove_top) else ("—", 0.0)
            b_txt, b_sc = bert_top[rank]  if rank < len(bert_top)  else ("—", 0.0)

            print(f"  #{rank+1}   "
                  f"{t_txt[:40]:<40} ({t_sc:.3f})  "
                  f"{g_txt[:40]:<40} ({g_sc:.3f})  "
                  f"{b_txt[:40]:<40} ({b_sc:.3f})")

    print("\nDone.")