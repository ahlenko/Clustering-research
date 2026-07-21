"""
config.py — усі налаштування проєкту.
"""
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Шляхи
    dataset_dir: str = "datasets"
    report_dir: str = "reports"
    log_file: str = "experiment.log"

    # Препроцесинг
    lang: str = "ukrainian"          
    remove_stopwords: bool = True
    lemmatize: bool = True

    # Список векторизаторів для експерименту
    vectorizers: List[str] = field(default_factory=lambda: [
        "tfidf",
        "word2vec",
        "fasttext",
        "doc2vec",
        "sentence_bert"
    ])

    # Список алгоритмів кластеризації
    cluster_algorithms: List[str] = field(default_factory=lambda: [
        "kmeans",
        "minibatch",
        "dbscan",
        "hierarchical",
        "birch",
        "spectral",
        "affinity",
        "hdbscan"
    ])

    # Параметри за замовчуванням
    n_clusters: int = 5

    # Бенчмарк
    runs_per_test: int = 3
    profile_memory: bool = True
    profile_cpu: bool = True

    # Візуалізація
    visualize: bool = True
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 15

    # Метрики
    internal_metrics: List[str] = field(default_factory=lambda: [
        "silhouette", "davies_bouldin", "calinski_harabasz"
    ])