"""
Module 6 Week B — Stretch: Embedding Space Explorer

Visualizes:
1. Word-level GloVe vectors (200 words, 5 categories) reduced to 2D
2. Document-level DistilBERT embeddings (20 BBC News articles) reduced to 2D

Dimensionality reduction: t-SNE (better for local cluster structure)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

# ── Reuse helpers from the lab ───────────────────────────────────────────────

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


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT using mean pooling."""
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

    last_hidden    = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    mask_expanded  = attention_mask.unsqueeze(-1).float()
    sum_hidden     = (last_hidden * mask_expanded).sum(dim=1)
    sum_mask       = mask_expanded.sum(dim=1).clamp(min=1e-9)
    mean_pooled    = (sum_hidden / sum_mask).squeeze(0)
    return mean_pooled.numpy()


def build_bert_matrix(texts, tokenizer, model):
    """Pre-compute DistilBERT embeddings for a list of texts."""
    embeddings = []
    total = len(texts)
    for i, text in enumerate(texts):
        embeddings.append(extract_bert_embedding(text, tokenizer, model))
        print(f"\r  BERT embeddings: {i + 1}/{total}", end="", flush=True)
    print()
    return np.array(embeddings, dtype=np.float32)


# ── Part 1: Word Embedding Visualization ────────────────────────────────────

# 200 words organized into 5 semantic categories.
# Categories chosen to represent clearly distinct semantic domains,
# which should appear as well-separated clusters in the 2D projection.

WORD_CATEGORIES = {
    "countries": [
        "france", "germany", "italy", "spain", "japan",
        "china", "brazil", "india", "canada", "australia",
        "russia", "mexico", "egypt", "nigeria", "argentina",
        "sweden", "norway", "denmark", "poland", "portugal",
        "greece", "turkey", "iran", "iraq", "pakistan",
        "thailand", "vietnam", "malaysia", "indonesia", "kenya",
        "ghana", "ethiopia", "colombia", "chile", "peru",
        "venezuela", "cuba", "ukraine", "romania", "hungary",
    ],
    "sports": [
        "football", "basketball", "tennis", "cricket", "rugby",
        "swimming", "cycling", "boxing", "golf", "volleyball",
        "hockey", "baseball", "skiing", "athletics", "gymnastics",
        "wrestling", "archery", "sailing", "rowing", "fencing",
        "marathon", "sprint", "tournament", "championship", "league",
        "stadium", "referee", "goalkeeper", "midfielder", "striker",
        "coach", "athlete", "medal", "trophy", "penalty",
        "goalkeeper", "defender", "forward", "pitcher", "batter",
        "serve", "forehand", "backhand", "dribble", "tackle",
    ],
    "technology": [
        "computer", "internet", "software", "hardware", "network",
        "algorithm", "database", "server", "browser", "keyboard",
        "processor", "memory", "storage", "wireless", "bluetooth",
        "encryption", "firewall", "bandwidth", "download", "upload",
        "smartphone", "tablet", "laptop", "desktop", "monitor",
        "camera", "microphone", "speaker", "router", "modem",
        "programming", "coding", "debugging", "framework", "library",
        "cloud", "streaming", "startup", "developer", "engineer",
        "interface", "pixel", "resolution", "semiconductor", "chip",
    ],
    "emotions": [
        "happy", "sad", "angry", "fearful", "surprised",
        "disgusted", "joyful", "anxious", "excited", "bored",
        "lonely", "proud", "ashamed", "guilty", "jealous",
        "hopeful", "frustrated", "calm", "nervous", "confused",
        "love", "hatred", "grief", "pleasure", "pain",
        "fear", "rage", "bliss", "sorrow", "delight",
        "anxiety", "depression", "happiness", "stress", "relief",
        "envy", "pity", "trust", "disgust", "contempt",
        "empathy", "compassion", "courage", "despair", "hope",
    ],
    "finance": [
        "bank", "stock", "market", "investment", "profit",
        "revenue", "budget", "tax", "loan", "debt",
        "mortgage", "inflation", "recession", "dividend", "equity",
        "bond", "fund", "portfolio", "asset", "liability",
        "currency", "exchange", "interest", "credit", "capital",
        "salary", "wage", "pension", "insurance", "audit",
        "merger", "acquisition", "startup", "venture", "shareholder",
        "bankruptcy", "liquidity", "deficit", "surplus", "subsidy",
        "tariff", "import", "export", "gdp", "unemployment",
    ],
}

# Color palette for categories
CATEGORY_COLORS = {
    "countries":  "#e74c3c",   # red
    "sports":     "#2ecc71",   # green
    "technology": "#3498db",   # blue
    "emotions":   "#f39c12",   # orange
    "finance":    "#9b59b6",   # purple
}

# Words to annotate on the plot (notable / interesting choices)
ANNOTATE_WORDS = [
    "france", "japan", "brazil",          # countries
    "football", "tennis", "marathon",     # sports
    "internet", "algorithm", "cloud",     # technology
    "love", "anger", "happiness",          # emotions (note: GloVe uses 'anger' not 'angry')
    "inflation", "stock", "bankruptcy",   # finance
]


def build_word_data(glove):
    """Build word list, category labels, and GloVe matrix for the 200 words."""
    words, labels, vectors = [], [], []

    for category, word_list in WORD_CATEGORIES.items():
        for word in word_list:
            if word in glove:
                words.append(word)
                labels.append(category)
                vectors.append(glove[word])
            else:
                print(f"  [OOV] '{word}' not in GloVe — skipping")

    print(f"  Words found in GloVe: {len(words)} / "
          f"{sum(len(v) for v in WORD_CATEGORIES.values())}")
    return words, labels, np.array(vectors, dtype=np.float32)


def plot_word_embeddings(words, labels, vectors_2d, save_path="word_embedding_plot.png"):
    """Create and save annotated 2D scatter plot of word embeddings."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    for category, color in CATEGORY_COLORS.items():
        mask = [l == category for l in labels]
        x = vectors_2d[mask, 0]
        y = vectors_2d[mask, 1]
        ax.scatter(x, y, c=color, label=category.capitalize(),
                   s=60, alpha=0.75, edgecolors="white", linewidths=0.5)

    # Annotate notable words
    label_array = np.array(labels)
    for word in ANNOTATE_WORDS:
        if word in words:
            idx = words.index(word)
            x, y = vectors_2d[idx]
            color = CATEGORY_COLORS[labels[idx]]
            ax.annotate(
                word,
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec=color, lw=0.8),
            )

    legend_patches = [
        mpatches.Patch(color=c, label=cat.capitalize())
        for cat, c in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10,
              framealpha=0.9, edgecolor="#cccccc")

    ax.set_title("GloVe Word Embeddings — t-SNE Projection (200 words, 5 categories)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ── Part 2: Document Embedding Visualization ─────────────────────────────────

BBC_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

BBC_CATEGORY_COLORS = {
    "business":      "#e74c3c",
    "entertainment": "#f39c12",
    "politics":      "#3498db",
    "sport":         "#2ecc71",
    "tech":          "#9b59b6",
}


def select_articles(df, n_per_category=4):
    """Select n articles per BBC category (total = 5 * n = 20)."""
    selected = []
    for cat in BBC_CATEGORIES:
        subset = df[df["category"] == cat].head(n_per_category)
        selected.append(subset)
        print(f"  {cat}: {len(subset)} articles selected")
    return pd.concat(selected, ignore_index=True)


def plot_document_embeddings(articles_df, vectors_2d,
                             save_path="document_embedding_plot.png"):
    """Create and save annotated 2D scatter plot of document embeddings."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    for cat in BBC_CATEGORIES:
        mask = articles_df["category"] == cat
        x = vectors_2d[mask.values, 0]
        y = vectors_2d[mask.values, 1]
        ax.scatter(x, y, c=BBC_CATEGORY_COLORS[cat], label=cat.capitalize(),
                   s=120, alpha=0.85, edgecolors="white", linewidths=1.0)

    # Annotate every article (short label)
    for i, row in articles_df.iterrows():
        local_idx = articles_df.index.get_loc(i)
        x, y = vectors_2d[local_idx]
        label_text = row["text"][:35].strip() + "…"
        color = BBC_CATEGORY_COLORS[row["category"]]
        ax.annotate(
            label_text,
            xy=(x, y),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.55,
                      ec=color, lw=0.6),
        )

    legend_patches = [
        mpatches.Patch(color=c, label=cat.capitalize())
        for cat, c in BBC_CATEGORY_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10,
              framealpha=0.9, edgecolor="#cccccc")

    ax.set_title(
        "DistilBERT Document Embeddings — t-SNE Projection (20 BBC News articles)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # ── Part 1: GloVe Word Embeddings ────────────────────────────────────
    print("\n[Part 1] Loading GloVe vectors...")
    glove = load_glove("data/glove_50k_50d.txt")
    print(f"  Loaded {len(glove):,} GloVe vectors")

    print("\n  Building word dataset (200 words, 5 categories)...")
    words, labels, word_matrix = build_word_data(glove)

    print("\n  Applying t-SNE to GloVe word vectors (50d → 2d)...")
    # Perplexity=30 works well for ~200 points (rule of thumb: n/7 to n/5).
    # t-SNE chosen over PCA because it preserves local cluster structure,
    # making semantic word groups visually distinct even when globally
    # the embedding space is not linearly separable.
    tsne_word = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        init="pca",         # PCA init gives more stable results
        learning_rate="auto",
    )
    word_2d = tsne_word.fit_transform(word_matrix)
    print(f"  t-SNE word output shape: {word_2d.shape}")

    print("\n  Plotting word embeddings...")
    plot_word_embeddings(words, labels, word_2d, "word_embedding_plot.png")

    # ── Part 2: DistilBERT Document Embeddings ───────────────────────────
    print("\n[Part 2] Loading BBC News corpus...")
    df = pd.read_csv("data/bbc_news.csv")
    print(f"  Total articles: {len(df)}")
    print(f"  Categories: {df['category'].unique().tolist()}")

    print("\n  Selecting 20 articles (4 per category)...")
    articles_df = select_articles(df, n_per_category=4)

    print("\n  Loading DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model     = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    print("\n  Computing DistilBERT embeddings for 20 articles...")
    doc_matrix = build_bert_matrix(
        articles_df["text"].tolist(), tokenizer, model
    )
    print(f"  Document embedding matrix shape: {doc_matrix.shape}")

    print("\n  Applying t-SNE to document vectors (768d → 2d)...")
    # Perplexity=5 for only 20 points — must be < n_samples.
    tsne_doc = TSNE(
        n_components=2,
        perplexity=5,
        max_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    doc_2d = tsne_doc.fit_transform(doc_matrix)
    print(f"  t-SNE document output shape: {doc_2d.shape}")

    print("\n  Plotting document embeddings...")
    plot_document_embeddings(articles_df, doc_2d, "document_embedding_plot.png")

    print("\n✓ Done! Outputs:")
    print("   word_embedding_plot.png")
    print("   document_embedding_plot.png")