"""
experiment.py – головний клас, що запускає всі комбінації
векторизатор-алгоритм-датасет, збирає метрики та зберігає звіт.
"""
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from config import Config
from core.dataset import TextDataset
from core.result import ExperimentResult
from core.benchmark import Benchmark
from preprocessing.pipeline import PreprocessingPipeline
from metrics.internal import evaluate_internal
from metrics.external import evaluate_external
from visualization.plots import plot_clusters

# Імпорт усіх векторизаторів
from vectorizers.tfidf import TfidfVectorizerAdapter
from vectorizers.word2vec import Word2VecVectorizer
from vectorizers.fasttext import FastTextVectorizer
from vectorizers.doc2vec import Doc2VecVectorizer
from vectorizers.sentence_bert import SentenceBERTVectorizer

# Імпорт усіх кластеризаторів
from clustering.kmeans import KMeansClusterer
from clustering.minibatch import MiniBatchKMeansClusterer
from clustering.dbscan import DBSCANClusterer
from clustering.hierarchical import HierarchicalClusterer
from clustering.birch import BirchClusterer
from clustering.spectral import SpectralClusterer
from clustering.affinity import AffinityPropagationClusterer
from clustering.hdbscan import HDBSCANClusterer

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Керує повним циклом дослідження."""
    def __init__(self, config: Config):
        self.config = config
        self.benchmark = Benchmark(
            runs=config.runs_per_test,
            profile_memory=config.profile_memory,
            profile_cpu=config.profile_cpu
        )
        self.results: list[ExperimentResult] = []
        # Створюємо теки, якщо треба
        Path(config.report_dir).mkdir(parents=True, exist_ok=True)

    def run_all(self):
        """Головний метод: обробляє всі датасети."""
        dataset_files = list(Path(self.config.dataset_dir).glob("*.csv"))
        if not dataset_files:
            logger.warning(f"Не знайдено CSV-файлів у {self.config.dataset_dir}")
            return

        for ds_path in dataset_files:
            logger.info(f"===== Датасет: {ds_path.name} =====")
            dataset = TextDataset(str(ds_path)).load()

            # Препроцесинг
            pipeline = PreprocessingPipeline(
                remove_stopwords=self.config.remove_stopwords,
                lemmatize=self.config.lemmatize,
                lang=self.config.lang
            )
            # Отримуємо списки токенів для кожного тексту
            tokenized_texts = [pipeline.process(text) for text in dataset.texts]

            # Для кожного векторизатора
            for vec_name in self.config.vectorizers:
                logger.info(f"-- Векторизатор: {vec_name} --")
                vectorizer = self._create_vectorizer(vec_name)
                # Вимірюємо векторизацію
                X, vec_stats = self.benchmark.measure(
                    vectorizer.fit_transform, tokenized_texts,
                    name=f"vec_{vec_name}"
                )
                logger.info(f"Статистика векторизації: {vec_stats}")

                # Для кожного алгоритму кластеризації
                for alg_name in self.config.cluster_algorithms:
                    logger.info(f"   Алгоритм: {alg_name}")
                    clusterer = self._create_clusterer(alg_name)

                    # Вимірюємо кластеризацію
                    labels, clust_stats = self.benchmark.measure(
                        clusterer.fit_predict, X,
                        name=f"clust_{alg_name}"
                    )
                    logger.info(f"Статистика кластеризації: {clust_stats}")

                    # Оцінка якості
                    internal = evaluate_internal(X, labels)
                    external = {}
                    if dataset.labels is not None:
                        external = evaluate_external(dataset.labels, labels)

                    # Збереження результату
                    result = ExperimentResult(
                        dataset=ds_path.stem,
                        vectorizer=vec_name,
                        algorithm=alg_name,
                        internal_metrics=internal,
                        external_metrics=external
                    )
                    self.results.append(result)

                    # Візуалізація (опціонально)
                    if self.config.visualize:
                        fname = f"{ds_path.stem}_{vec_name}_{alg_name}.png"
                        save_path = Path(self.config.report_dir) / fname
                        plot_clusters(X, labels,
                                      title=f"{vec_name} + {alg_name}",
                                      save_path=str(save_path),
                                      method='tsne',
                                      perplexity=self.config.tsne_perplexity)

        # Збереження загального звіту
        self._save_report()

    def _create_vectorizer(self, name: str):
        """Фабрика векторизаторів."""
        if name == "tfidf":
            return TfidfVectorizerAdapter()
        elif name == "word2vec":
            return Word2VecVectorizer()
        elif name == "fasttext":
            return FastTextVectorizer()
        elif name == "doc2vec":
            return Doc2VecVectorizer()
        elif name == "sentence_bert":
            return SentenceBERTVectorizer()
        else:
            raise ValueError(f"Невідомий векторизатор: {name}")

    def _create_clusterer(self, name: str):
        """Фабрика алгоритмів кластеризації."""
        n = self.config.n_clusters
        if name == "kmeans":
            return KMeansClusterer(n_clusters=n)
        elif name == "minibatch":
            return MiniBatchKMeansClusterer(n_clusters=n)
        elif name == "dbscan":
            return DBSCANClusterer()
        elif name == "hierarchical":
            return HierarchicalClusterer(n_clusters=n)
        elif name == "birch":
            return BirchClusterer(n_clusters=n)
        elif name == "spectral":
            return SpectralClusterer(n_clusters=n)
        elif name == "affinity":
            return AffinityPropagationClusterer()
        elif name == "hdbscan":
            return HDBSCANClusterer()
        else:
            raise ValueError(f"Невідомий алгоритм: {name}")

    def _save_report(self):
        """Зберігає всі результати в CSV."""
        if not self.results:
            logger.warning("Немає результатів для збереження.")
            return
        df = pd.DataFrame([r.to_dict() for r in self.results])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(self.config.report_dir) / f"full_report_{timestamp}.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"Підсумковий звіт збережено: {report_path}")