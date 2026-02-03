import os
import json
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from matplotlib.lines import Line2D

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

from gensim import corpora
from sentence_transformers import SentenceTransformer
import hdbscan
import umap.umap_ as umap
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Global Configurations ---

CUSTOM_STOPWORDS = set(stopwords.words('english')).union({
    'opens', 'window', 'new', 'openai', 'openais', 'gpt', 'says', 'via', 'us',
    'also', 'use', 'using', 'example', 'time', 'one', 'make', 'may', 'well',
    'many', 'way', 'used', 'weve', 'including', 'across', 'text', 'tool',
    'content', 'result', 'like', 'would', 'different', 'question', 'set',
    'need', 'could', 'first', 'even', 'atthis', 'http', 'urlopens', 'https',
    'www', 'arxiv'
})

def parse_args():
    parser = argparse.ArgumentParser(description="AI Ethics Topic Clustering Pipeline")
    
    # Mode Selection
    parser.add_argument('--mode', type=str, choices=['pdf', 'article'], default='pdf',
                        help="Data source type: 'pdf' for directory of txt, 'article' for json")
    parser.add_argument('--filter_clusters', action='store_true', help="Discard specific clusters based on hardcoded IDs")
    parser.add_argument('--merge_topics', action='store_true', default=True, help="Merge small clusters into meta-topics")
    
    # HDBSCAN Hyperparams
    parser.add_argument('--min_cluster_size', type=int, default=5, help="HDBSCAN min_cluster_size")
    parser.add_argument('--min_samples', type=int, default=1, help="HDBSCAN min_samples")
    
    # UMAP Hyperparams
    parser.add_argument('--n_neighbors', type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument('--min_dist', type=float, default=0.03, help="UMAP min_dist")
    
    # Paths
    parser.add_argument('--data_dir', type=str, default="../../cleaned_texts", help="Directory for txt files")
    parser.add_argument('--json_path', type=str, default="../../Cleaned/OpenAI/OpenAI_final_processed_data.json")
    parser.add_argument('--keywords_path', type=str, default="../../keywords_ai_ethics.json")
    
    return parser.parse_args()

# --- Data Loading & Preprocessing ---

def load_text_files(root_directory):
    """Recursively loads .txt files starting with YYYY prefix."""
    year_data = []
    if not os.path.isdir(root_directory):
        print(f"Error: Directory '{root_directory}' not found.")
        return []

    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.lower().endswith(".txt") and filename[:4].isdigit():
                year = '20' + filename[:2]
                try:
                    with open(os.path.join(dirpath, filename), 'r', encoding='utf-8') as f:
                        year_data.append({'Date': year, 'Article': f.read()})
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    return year_data

def clean_text(text, lemmatizer):
    """Cleans text, removes stopwords, and generates unigrams + bigrams."""
    text = re.sub(r'[^a-z\s]', '', text.lower())
    tokens = [word for word in text.split() if word not in CUSTOM_STOPWORDS and len(word) > 2]
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]
    
    # Generate bigrams
    bigrams = ['_'.join(bg) for bg in ngrams(lemmas, 2)]
    return lemmas + bigrams

# --- Visualization Helpers ---

def set_tight_limits(ax, XY, q=0.02, pad=0.06):
    """Trims extreme outliers for a cleaner visualization."""
    x0, x1 = np.quantile(XY[:, 0], [q, 1 - q])
    y0, y1 = np.quantile(XY[:, 1], [q, 1 - q])
    dx, dy = (x1 - x0), (y1 - y0)
    ax.set_xlim(x0 - pad * dx, x1 + pad * dx)
    ax.set_ylim(y0 - pad * dy, y1 + pad * dy)

# --- Main Logic ---

def main():
    args = parse_args()
    lemmatizer = WordNetLemmatizer()

    # 1. Load Data
    if args.mode == 'pdf':
        data = load_text_files(args.data_dir)
    else:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
    
    articles = [item['Article'] for item in data if item.get('Article')]
    cleaned_tokens_list = [clean_text(art, lemmatizer) for art in articles]

    # 2. Extract Ethics-related N-grams
    with open(args.keywords_path, "r", encoding="utf-8") as f:
        keyword_concepts = json.load(f)
    
    flat_keywords = [lemmatizer.lemmatize(kw) for k_list in keyword_concepts.values() for kw in k_list]
    
    # TF-IDF to find relevant n-grams
    corpus_docs = [' '.join(t) for t in cleaned_tokens_list]
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), min_df=3)
    X_tfidf = vectorizer.fit_transform(corpus_docs)
    
    feature_names = vectorizer.get_feature_names_out()
    ethics_indices = [i for i, ng in enumerate(feature_names) if any(k in ng for k in flat_keywords)]
    ngram_texts = [feature_names[i] for i in ethics_indices]

    # 3. Embedding & Clustering
    print("Encoding with BERT...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(ngram_texts)

    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        prediction_data=True
    ).fit(embeddings)
    
    # Handle noise by assigning to highest probability cluster
    labels = clusterer.labels_
    if -1 in labels:
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
        for i in np.where(labels == -1)[0]:
            labels[i] = np.argmax(soft_clusters[i])

    # 4. Merging Logic (Referencing your defined maps)
    # Note: In a production setting, these maps should be in a config file
    discard_ids = {1, 2, 3} if args.mode == 'pdf' else {0, 1} # Simplified for example
    
    # Create mask for filtering
    plot_mask = np.array([lbl not in discard_ids for lbl in labels])
    filtered_labels = labels[plot_mask]

    # 5. Dimensionality Reduction (UMAP)
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=42)
    X_embedded = reducer.fit_transform(embeddings)[plot_mask]

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    cmap = plt.cm.get_cmap("tab20", len(set(filtered_labels)))
    
    for i, lbl in enumerate(sorted(set(filtered_labels))):
        mask = (filtered_labels == lbl)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=f"Cluster {lbl}", s=50, alpha=0.8)

    set_tight_limits(ax, X_embedded)
    ax.set_title(f"Clustering Results - {args.mode.upper()}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_name = f"results_{args.mode}.png"
    plt.savefig(output_name, bbox_inches="tight")
    print(f"Saved plot to {output_name}")

if __name__ == "__main__":
    main()